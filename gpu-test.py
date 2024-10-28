import time
import torch
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer (using GPT-2 model)
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

# Prepare input data
inputs = tokenizer("This is a test sentence.", return_tensors="pt").to("cuda")

# Function to measure throughput and latency
def measure_throughput_and_latency(model, inputs, num_iterations=100):
    latencies = []
    start_time = time.time()
    
    for _ in range(num_iterations):
        start_iteration = time.time()
        # Setting pad_token_id explicitly to avoid warnings
        outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
        end_iteration = time.time()
        
        latencies.append(end_iteration - start_iteration)

    total_time = time.time() - start_time
    throughput = num_iterations / total_time  # Calculate throughput (requests per second)
    avg_latency = sum(latencies) / len(latencies)  # Calculate average latency

    return throughput, avg_latency

# Function to measure memory usage
def measure_memory_usage(model, inputs):
    torch.cuda.reset_peak_memory_stats()  # Reset memory statistics before measurement

    # Run inference
    outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)

    # Get GPU memory usage
    memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Peak memory usage in MB

    return memory_allocated, peak_memory

# Function to measure GPU utilization
def measure_gpu_utilization():
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
    gpu_util, memory_util = result.stdout.decode('utf-8').strip().split(',')
    
    return float(gpu_util.strip()), float(memory_util.strip())

# Function to run the full performance test
def full_performance_test(model, inputs, num_iterations=100):
    throughput, avg_latency = measure_throughput_and_latency(model, inputs, num_iterations)
    memory_allocated, peak_memory = measure_memory_usage(model, inputs)
    gpu_util, memory_util = measure_gpu_utilization()

    return {
        "throughput": throughput,
        "avg_latency": avg_latency,
        "memory_allocated": memory_allocated,
        "peak_memory": peak_memory,
        "gpu_utilization": gpu_util,
        "memory_utilization": memory_util
    }

# Run the performance test 10 times and calculate the average
def run_multiple_tests(model, inputs, num_iterations=100, num_tests=10):
    results_sum = {
        "throughput": 0,
        "avg_latency": 0,
        "memory_allocated": 0,
        "peak_memory": 0,
        "gpu_utilization": 0,
        "memory_utilization": 0
    }

    for i in range(num_tests):
        print(f"\n=== Running Test {i+1} ===")
        results = full_performance_test(model, inputs, num_iterations)
        
        for key in results:
            results_sum[key] += results[key]
    
    # Calculate average values over the number of tests
    results_avg = {key: value / num_tests for key, value in results_sum.items()}
    
    return results_avg

# Execute the test 10 times and calculate the average
average_results = run_multiple_tests(model, inputs, num_iterations=100, num_tests=10)

# Output the average results
print("\n=== Average Performance Test Results ===")
for key, value in average_results.items():
    print(f"{key}: {value}")
