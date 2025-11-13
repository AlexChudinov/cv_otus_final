from itertools import islice

def chunked_iter(iterable, chunk_size=100):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, chunk_size))
        if not chunk:
            return
        yield chunk
