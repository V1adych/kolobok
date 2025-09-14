import time
from statistics import mean, stdev

from tire_vision.text.index.db import TireModelDatabase
from tire_vision.config import TireAnnotationPipelineConfig

def get_time(db: TireModelDatabase, queries: list[str]):
    start_time = time.time()
    db.query(queries)
    end_time = time.time()
    return end_time - start_time

def benchmark(db: TireModelDatabase, queries: list[str], n_runs: int = 10):
    times = []
    for _ in range(n_runs):
        times.append(get_time(db, queries))
    return mean(times), stdev(times)

def main():
    cfg = TireAnnotationPipelineConfig()
    index_cfg = cfg.index_config
    db = TireModelDatabase(index_cfg)

    queries = [
        "PLANET",
        "PLANET DC",
        "NOKIAN",
        "TIRES",
        "VIATTI",
        "MICHELIN",
        "GREENMAX",
        "205/55R16",
        "205/55R16 91H",
        "KAMA",
        "217",
        "KAMA EURO",
        "KAMA-217"
    ]

    print(benchmark(db, queries))

if __name__ == "__main__":
    main()