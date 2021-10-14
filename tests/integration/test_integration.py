from jina import Flow


def test_use_in_flow(mixed_docs):
    with Flow.load_config('flow.yml') as flow:
        data = flow.post(
            on='/index', inputs=mixed_docs, return_results=True
        )
        for doc in data[0].docs:
            assert doc.blob is not None
            assert doc.blob.shape == (1024, 3)
