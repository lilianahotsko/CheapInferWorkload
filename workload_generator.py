#!/usr/bin/env python3
import json
import random
import argparse
import yaml
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import tiktoken


@dataclass
class Request:
    """Represents a single LLM request"""
    request_id: int
    prompt: str
    expected_completion_tokens: int
    arrival_time: float
    prompt_tokens: int = 0
    metadata: Dict[str, Any] = None
    
    def to_dict(self, include_metadata: bool = True) -> Dict:
        result = {
            "request_id": self.request_id,
            "prompt": self.prompt,
            "expected_completion_tokens": self.expected_completion_tokens,
            "arrival_time": self.arrival_time
        }
        if include_metadata:
            result.update({
                "prompt_tokens": self.prompt_tokens,
                "metadata": self.metadata or {}
            })
        return result


class TokenCounter:
    """Handles token counting for prompts and completions"""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base encoding
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text"""
        return len(self.encoding.encode(text))
    
    def estimate_tokens(self, text: str) -> int:
        """Rough estimation: ~4 chars per token"""
        return len(text) // 4


class DatasetLoader:
    """Loads and manages multiple workload datasets"""
    
    def __init__(self, token_counter: TokenCounter):
        self.token_counter = token_counter
        self.datasets = {}
    
    def load_dataset(self, name: str, path: str) -> List[Dict]:
        """Load a dataset from a JSONL file"""
        data = []
        file_path = Path(path)
        
        if not file_path.exists():
            print(f"Warning: Dataset {name} not found at {path}")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError:
                        continue
        
        print(f"Loaded {len(data)} items from {name}")
        return data
    
    def load_all(self, data_sources: List[Dict]) -> Dict[str, List[Dict]]:
        """Load all configured datasets"""
        for source in data_sources:
            if source.get('enabled', True):
                name = source['name']
                path = source['path']
                self.datasets[name] = {
                    'data': self.load_dataset(name, path),
                    'weight': source.get('weight', 1.0)  # Default weight if not specified
                }
        return self.datasets
    
    def get_item_tokens(self, item: Dict) -> Tuple[int, int]:
        prompt = item.get('prompt', item.get('instruction', item.get('text', '')))
        completion = item.get('completion', item.get('output', item.get('response', '')))
        prompt_tokens = self.token_counter.count_tokens(prompt) if prompt else 0
        completion_tokens = self.token_counter.count_tokens(completion) if completion else 0
        
        return prompt_tokens, completion_tokens


class WorkloadGenerator:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        seed = self.config.get('random_seed')
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.token_counter = TokenCounter()
        self.loader = DatasetLoader(self.token_counter)
        self.requests = []
    
    def load_datasets(self):
        """Load all configured datasets"""
        self.datasets = self.loader.load_all(self.config['data_sources'])
    
    def sample_dataset_item(self, 
                           prompt_range: Tuple[int, int],
                           completion_range: Tuple[int, int]) -> Dict:

        all_items = []
        for name, dataset_info in self.datasets.items():
            if dataset_info['data']:
                all_items.extend(dataset_info['data'])
        
        if not all_items:
            raise ValueError("No datasets available")
        
        max_attempts = 100
        for _ in range(max_attempts):
            item = random.choice(all_items)
            prompt_tokens, completion_tokens = self.loader.get_item_tokens(item)
            
            prompt_min, prompt_max = prompt_range
            completion_min, completion_max = completion_range
            
            if (prompt_min <= prompt_tokens <= prompt_max and 
                completion_min <= completion_tokens <= completion_max):
                return item, prompt_tokens, completion_tokens
        
        item = random.choice(all_items)
        prompt_tokens, completion_tokens = self.loader.get_item_tokens(item)
        
        return item, prompt_tokens, completion_tokens
    
    def generate_arrival_times(self, num_requests: int) -> List[float]:
        """Generate arrival times based on configured pattern"""
        pattern_config = self.config['arrival_pattern']
        pattern_type = pattern_config['type']
        start_time = self.config['advanced']['timestamp_start']
        
        if pattern_type == 'poisson':
            rate = pattern_config['rate']
            inter_arrival_times = np.random.exponential(1.0 / rate, num_requests)
            arrival_times = np.cumsum(inter_arrival_times) + start_time
            
        elif pattern_type == 'uniform':
            rate = pattern_config['rate']
            interval = 1.0 / rate
            arrival_times = np.arange(num_requests) * interval + start_time
            
        elif pattern_type == 'burst':
            burst_config = pattern_config['burst']
            duration = burst_config['duration']
            interval = burst_config['interval']
            multiplier = burst_config['multiplier']
            base_rate = pattern_config['rate']
            
            arrival_times = []
            current_time = start_time
            requests_generated = 0
            
            while requests_generated < num_requests:
                burst_rate = base_rate * multiplier
                burst_requests = int(duration * burst_rate)
                for _ in range(min(burst_requests, num_requests - requests_generated)):
                    arrival_times.append(current_time)
                    current_time += np.random.exponential(1.0 / burst_rate)
                    requests_generated += 1
                    if requests_generated >= num_requests:
                        break
                
                current_time += interval - duration
                normal_requests = int((interval - duration) * base_rate)
                for _ in range(min(normal_requests, num_requests - requests_generated)):
                    arrival_times.append(current_time)
                    current_time += np.random.exponential(1.0 / base_rate)
                    requests_generated += 1
                    if requests_generated >= num_requests:
                        break
            
            arrival_times = np.array(arrival_times[:num_requests])
            
        else: 
            rate = pattern_config.get('rate', 1.0)
            interval = 1.0 / rate
            arrival_times = np.arange(num_requests) * interval + start_time
        
        return arrival_times.tolist()
    
    def generate_workload(self) -> List[Request]:
        """Generate the complete workload"""
        num_requests = self.config['output']['num_requests']
        
        prompt_dist = self.config['prompt_distribution']
        completion_dist = self.config['completion_distribution']
        
        sampling_plan = []
        for p_category, p_config in prompt_dist.items():
            for c_category, c_config in completion_dist.items():
                count = int(num_requests * 
                          (p_config['percentage'] / 100.0) * 
                          (c_config['percentage'] / 100.0))
                
                prompt_range = (p_config['min_tokens'], p_config['max_tokens'])
                completion_range = (c_config['min_tokens'], c_config['max_tokens'])
                
                sampling_plan.append({
                    'count': count,
                    'prompt_range': prompt_range,
                    'completion_range': completion_range,
                    'categories': (p_category, c_category)
                })
        
        # Adjust for rounding errors
        total_count = sum(item['count'] for item in sampling_plan)
        if total_count < num_requests:
            sampling_plan[0]['count'] += (num_requests - total_count)
        
        # Generate requests
        requests = []
        request_id = 0
        
        for plan in sampling_plan:
            for _ in range(plan['count']):
                item, prompt_tokens, completion_tokens = self.sample_dataset_item(
                    plan['prompt_range'],
                    plan['completion_range']
                )
                
                # Extract prompt
                prompt = item.get('prompt', item.get('instruction', item.get('text', '')))
                
                # Get expected completion tokens
                expected_tokens = completion_tokens if completion_tokens > 0 else \
                                random.randint(*plan['completion_range'])
                
                request = Request(
                    request_id=request_id,
                    prompt=prompt,
                    expected_completion_tokens=expected_tokens,
                    arrival_time=0.0,  # Will be set later
                    prompt_tokens=prompt_tokens,
                    metadata={
                        'prompt_category': plan['categories'][0],
                        'completion_category': plan['categories'][1]
                    }
                )
                
                requests.append(request)
                request_id += 1
        
        # Generate and assign arrival times
        arrival_times = self.generate_arrival_times(len(requests))
        for request, arrival_time in zip(requests, arrival_times):
            request.arrival_time = arrival_time
        
        # Sort by arrival time
        requests.sort(key=lambda x: x.arrival_time)
        
        # Filter duplicates if requested
        if self.config['advanced']['filter_duplicates']:
            seen_prompts = set()
            filtered_requests = []
            for request in requests:
                if request.prompt not in seen_prompts:
                    seen_prompts.add(request.prompt)
                    filtered_requests.append(request)
            requests = filtered_requests
        
        if self.config['advanced']['shuffle_output']:
            random.shuffle(requests)
            requests.sort(key=lambda x: x.arrival_time)
        
        return requests
    
    def save_workload(self, requests: List[Request]):
        output_file = self.config['output']['file']
        include_metadata = self.config['advanced']['include_metadata']
        
        output_data = {
            'metadata': {
                'total_requests': len(requests),
                'config': self.config
            },
            'requests': [asdict(r) for r in requests]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nWorkload saved to {output_file}")
        print(f"Total requests: {len(requests)}")
    
    def print_statistics(self, requests: List[Request]):
        print("\n" + "="*60)
        print("Workload Statistics")
        print("="*60)
        
        total = len(requests)
        prompt_tokens = [r.prompt_tokens for r in requests]
        completion_tokens = [r.expected_completion_tokens for r in requests]
        
        print(f"\nTotal Requests: {total}")
        print(f"\nPrompt Tokens:")
        print(f"  Min: {min(prompt_tokens)}")
        print(f"  Max: {max(prompt_tokens)}")
        print(f"  Mean: {np.mean(prompt_tokens):.2f}")
        print(f"  Median: {np.median(prompt_tokens):.2f}")
        
        print(f"\nCompletion Tokens:")
        print(f"  Min: {min(completion_tokens)}")
        print(f"  Max: {max(completion_tokens)}")
        print(f"  Mean: {np.mean(completion_tokens):.2f}")
        print(f"  Median: {np.median(completion_tokens):.2f}")
        
        categories = defaultdict(int)
        for r in requests:
            if r.metadata:
                key = f"{r.metadata.get('prompt_category', 'unknown')}-{r.metadata.get('completion_category', 'unknown')}"
                categories[key] += 1
        
        print(f"\nCategory Distribution:")
        for category, count in sorted(categories.items()):
            print(f"  {category}: {count} ({100*count/total:.1f}%)")
        
        if len(requests) > 1:
            arrival_times = [r.arrival_time for r in requests]
            print(f"\nArrival Times:")
            print(f"  First: {min(arrival_times):.2f}s")
            print(f"  Last: {max(arrival_times):.2f}s")
            print(f"  Duration: {max(arrival_times) - min(arrival_times):.2f}s")


def main():
    parser = argparse.ArgumentParser(description='Generate LLM workload for testing')
    parser.add_argument('--config', '-c', default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--output', '-o', 
                       help='Override output file path')
    
    args = parser.parse_args()
    
    print("="*60)
    print("LLM Workload Generator")
    print("="*60)
    print(f"\nLoading configuration from {args.config}...")
    generator = WorkloadGenerator(args.config)
    if args.output:
        generator.config['output']['file'] = args.output

    print("\nLoading datasets...")
    generator.load_datasets()
    print("\nGenerating workload...")
    requests = generator.generate_workload()
    generator.print_statistics(requests)
    generator.save_workload(requests)
    print("\n" + "="*60)
    print("Generation complete!")
    print("="*60)


if __name__ == '__main__':
    main()

