from mangum import Mangum

from pidetect.api import app
from pidetect.classifier import get_classifier

get_classifier()

handler = Mangum(app, lifespan="off")
