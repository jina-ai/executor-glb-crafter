from operator import itemgetter

import pytest
import requests
from jina import Document, DocumentArray, Executor
from glb_crafter import GlbCrafter


@pytest.fixture
def default_crafter():
    return GlbCrafter()


@pytest.fixture
def mixed_docs():
    file_url = 'https://storage.googleapis.com/showcase-3d-models/{0}'
    list_url = 'https://storage.googleapis.com/storage/v1/b/showcase-3d-models/o'
    data = requests.get(list_url).json()
    filenames = list(map(itemgetter('name'), data['items']))[:10]
    da = DocumentArray()
    for i, filename in enumerate(filenames):
        if i%2:
            da.append(Document(uri=file_url.format(filename.replace(' ', '%20'))))
        else:
            doc = Document(uri=file_url.format(filename.replace(' ', '%20')))
            doc.convert_uri_to_buffer()
            da.append(doc)
    return da


@pytest.fixture
def document_uri():
    return Document(uri='https://storage.googleapis.com/showcase-3d-models/ShapeNetV2/airplane_aeroplane_plane_0.glb')


@pytest.fixture
def document_blob(document_uri):
    d = Document(document_uri, copy=True)
    d.convert_uri_to_buffer()
    return d


def test_empty_docs(default_crafter):
    da = DocumentArray()
    default_crafter.craft(da)
    assert len(da) == 0


def test_input_none(default_crafter):
    default_crafter.craft(None)


def test_craft_uri(default_crafter, document_uri):
    default_crafter.craft(DocumentArray([document_uri]))
    assert document_uri.blob is not None
    assert document_uri.blob.shape == (2048, 3)


def test_craft_blob(default_crafter, document_blob):
    default_crafter.craft(DocumentArray([document_blob]))
    assert document_blob.blob is not None
    assert document_blob.blob.shape == (2048, 3)


@pytest.mark.parametrize('n_samples', (1024, 2048, 4096))
def test_n_samples(n_samples, document_uri):
    crafter = GlbCrafter(n_samples=n_samples)
    crafter.craft(DocumentArray(document_uri))
    assert document_uri.blob is not None
    assert document_uri.blob.shape == (n_samples, 3)


def test_mixed_da(default_crafter, mixed_docs):
    default_crafter.craft(mixed_docs)
    for doc in mixed_docs:
        assert doc.blob is not None
        assert doc.blob.shape == (2048, 3)
