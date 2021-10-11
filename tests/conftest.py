from operator import itemgetter

import pytest
import requests
from jina import DocumentArray, Document


@pytest.fixture(scope='function')
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