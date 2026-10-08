import multiprocessing
from concurrent.futures import ThreadPoolExecutor

from integrations.foul_play_processes import SearchPool


def process_kind():
    return multiprocessing.get_start_method()


def test_search_pool_spawns_from_background_thread_and_reuses_workers():
    pool = SearchPool()

    def search():
        with pool.context(max_workers=1) as executor:
            return executor.submit(process_kind).result(timeout=20), executor

    try:
        with ThreadPoolExecutor() as thread:
            first = thread.submit(search).result(timeout=25)
            second = thread.submit(search).result(timeout=25)
        assert first[0] == second[0] == "spawn"
        assert first[1] is second[1]
    finally:
        pool.close()
