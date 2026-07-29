import requests
import time
import statistics

def benchmark_query(url, question, iterations=5):
    print(f"Iniciando benchmark para: '{question}'\n")
    latencies = []
    
    for i in range(iterations):
        start = time.perf_counter()
        response = requests.post(
            f"{url}/query",
            json={"question": question, "top_k": 3}
        )
        end = time.perf_counter()
        
        if response.status_code == 200:
            latencies.append((end - start) * 1000)
            print(f"Iteração {i+1}: {latencies[-1]:.2f} ms")
        else:
            print(f"Erro na iteração {i+1}: {response.status_code}")

    print("\n" + "="*30)
    print("RESULTADOS DO BENCHMARK")
    print("="*30)
    print(f"Média:   {statistics.mean(latencies):.2f} ms")
    print(f"Mediana: {statistics.median(latencies):.2f} ms")
    print(f"Mínimo:  {min(latencies):.2f} ms")
    print(f"Máximo:  {max(latencies):.2f} ms")

if __name__ == "__main__":
    # Certifique-se de que o servidor está rodando
    API_URL = "http://localhost:8000"
    TEST_QUERY = "Explain speculative decoding speedup."
    benchmark_query(API_URL, TEST_QUERY)
