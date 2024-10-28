# GPU Performance Testing with LLM Use Cases

Here’s the updated version of your code that will run the performance test 10 times and calculate the average of the performance metrics (throughput, latency, memory usage, and GPU utilization) across those runs.

### Code Explanation:
1. **`run_multiple_tests()`**: This function runs the performance test multiple times (10 times by default) and calculates the sum of the metrics over each test. After all tests are complete, it computes the average for each metric.
   
2. **Averaging Results**:
   - After 10 tests, the sum of each metric is divided by 10 to compute the average. This helps in smoothing out the variability in the test results.
   
3. **Loop to Run 10 Tests**:
   - The code runs the performance test 10 times. After each test, the results are accumulated and stored in `results_sum`.

4. **Final Average Output**:
   - After all tests are complete, the code prints the average values for throughput, latency, memory usage, and GPU utilization.

### Expected Output:
- The script will run the performance test 10 times and then display the average values for:
  - **Throughput** (requests per second)
  - **Average Latency** (time per request)
  - **Memory Allocated** (GPU memory allocated during inference)
  - **Peak Memory** (the highest GPU memory usage during the test)
  - **GPU Utilization** (percentage)
  - **Memory Utilization** (percentage)

### Test Method Overview:

The provided code is designed to measure the **performance of a GPT-2 model** on a GPU by testing key metrics over multiple runs (10 runs by default) and then calculating the **average** performance values. The method focuses on **four main performance metrics**: throughput, latency, memory usage, and GPU utilization.

### Key Metrics Tested:
1. **Throughput**: The number of requests (inferences) the model can process per second.
2. **Latency**: The average time taken for a single request (inference).
3. **Memory Usage**: The amount of GPU memory allocated and the peak memory usage during inference.
4. **GPU Utilization**: How much of the GPU's processing power is being used during the test.
5. **Memory Utilization**: The percentage of GPU memory that is in use during the test.

### Test Method Details:

1. **Multiple Runs for Accuracy**:
   - The performance test is repeated **10 times** to account for small fluctuations in GPU processing times and system load.
   - By running the test multiple times and averaging the results, we get a more reliable and accurate measure of the model’s performance.

2. **Steps in Each Test**:
   Each test iteration involves the following steps:

   - **Throughput and Latency Measurement**: 
     - The model performs **100 inferences** (controlled by the `num_iterations` variable), where each inference generates a response to the input text using the `generate()` function of the model.
     - The **throughput** is calculated as the number of requests processed per second, i.e., the total number of inferences divided by the total time taken.
     - The **latency** is the average time taken for one inference, calculated by recording the time for each inference and then averaging it over all iterations.

   - **Memory Usage Measurement**:
     - The **allocated memory** is the amount of GPU memory in use during the test.
     - The **peak memory usage** measures the highest amount of GPU memory used during inference.

   - **GPU Utilization**:
     - The GPU’s **processing utilization** is measured using the `nvidia-smi` command. This command returns the percentage of the GPU's processing power used during the test.
     - **Memory utilization** refers to the percentage of GPU memory in use during the test.

3. **Accumulating Results**:
   - For each test run, the results for all performance metrics (throughput, latency, memory, etc.) are summed.
   - After all 10 test runs, the code calculates the **average value** for each performance metric by dividing the summed values by 10.

4. **Output of Average Results**:
   - Once the 10 test runs are completed, the code prints the **average values** for each performance metric. This gives a better overall view of the model's performance on the GPU by accounting for potential variability in individual runs.

### Advantages of This Test Method:

1. **Increased Accuracy**:
   - By performing multiple runs and averaging the results, this method minimizes the effects of any outliers or variability caused by system load, GPU fluctuations, or background processes. The average values give a more stable and reliable measure of the model's actual performance.

2. **Comprehensive Performance Evaluation**:
   - This method evaluates all key aspects of the model's performance on a GPU, including both **inference speed** (throughput and latency) and **resource usage** (memory and GPU utilization). These are crucial metrics for understanding the practical performance of the model in production environments.

3. **Scalability**:
   - The test can be easily modified to run more or fewer iterations or to test larger models by changing the `num_iterations` or model parameters. This makes the method versatile for testing other models or configurations as needed.

The test method consists of running a **performance test 10 times** for a **GPT-2 model** on a GPU, measuring key metrics like throughput, latency, memory usage, and GPU utilization. After completing all test runs, the code averages the results, providing a stable and accurate measure of the model's overall performance. This method ensures that any variability in the individual tests is accounted for and provides a more reliable performance assessment.

![downloading_files](https://raw.githubusercontent.com/BrandonKang/gpu-llm-test/refs/heads/main/github_gpu_test.jpg)

The message you see when running the code indicates that various model files and tokenizer components are being downloaded. Here’s a breakdown of what each part means:

1. **Downloading Tokenizer and Model Files**:
   - **tokenizer_config.json**: This file contains the configuration settings for the tokenizer, specifying how the tokenizer should process input text.
   - **config.json**: This file contains the configuration of the GPT-2 model itself, including model architecture details like the number of layers, hidden size, etc.
   - **vocab.json**: This file holds the vocabulary that the tokenizer uses. It maps words or sub-words to their respective token IDs.
   - **merges.txt**: This file is used by tokenizers that rely on **byte pair encoding (BPE)** to split or merge words into tokens.
   - **tokenizer.json**: This is a compiled version of the tokenizer configuration that includes the vocab and merges in a more efficient format.
   - **model.safetensors**: This is the file containing the weights of the pre-trained GPT-2 model in a safe tensor format, which is used for inference and processing input data.
   - **generation_config.json**: This file includes the settings used by the model during text generation, like default `max_length` or `temperature`.

2. **Progress Bars** (`100%|███████████████████████|`):
   - Each line shows the download progress for these files. The bars indicate that the files are being downloaded from a remote source (likely from the **Hugging Face Model Hub**).
   - The `100%` means that the download is complete for each file.

3. **Download Speeds**:
   - The numbers after the download progress (e.g., `[00:00<00:00, 205kB/s]`) indicate how long the download took and the speed at which each file was downloaded. This information helps to monitor download performance.

### Why Does This Happen?
- When you use Hugging Face's `transformers` library to load a pre-trained model (e.g., GPT-2), it needs to download these files the first time it is run. These files include the model weights and tokenizer configuration, which are necessary for using the model.
- Once downloaded, these files are typically cached on your local system. So, the next time you run the script, the library will use the cached versions instead of downloading them again, speeding up the initialization process.

### Where Are These Files Downloaded From?
- These files are downloaded from **Hugging Face's Model Hub**. Hugging Face hosts many pre-trained models and tokenizers, allowing users to easily download and use them for various NLP tasks.
- The library uses URLs embedded in the model's repository to fetch these files directly to your local environment.

### What Happens After Downloading?
- After downloading, the `transformers` library uses these files to initialize the GPT-2 model and tokenizer.
- The **tokenizer** processes input text into a format suitable for the model, and the **model weights** allow the GPT-2 model to perform text generation tasks based on the pre-trained parameters.

This process is normal when using pre-trained models from Hugging Face, especially when running the model for the first time on a new system or environment.



