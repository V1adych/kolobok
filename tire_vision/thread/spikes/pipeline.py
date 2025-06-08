import numpy as np
import cv2
import torch

from tire_vision.thread.spikes.models import get_spike_counter, get_spike_classifier
from tire_vision.config import SPIKE_COUNTER_CHECKPOINT, SPIKE_CLASSIFIER_CHECKPOINT



class SpikePipeline:
    def __init__(self):
        self.counter = get_spike_counter(SPIKE_COUNTER_CHECKPOINT)
        self.classifier = get_spike_classifier(SPIKE_CLASSIFIER_CHECKPOINT)
