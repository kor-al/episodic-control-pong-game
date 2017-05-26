import sys
import time
import os
import logging

import numpy as np
import tensorflow as tf

from trainer import VAETrainer


# Experiment description.
GAME = "pong"
MODEL = "EC"
VERSION = 0  # Bump this for each new experiment.

EXPERIMENT_PATH = os.path.join("/experiment/acid", GAME, MODEL, str(VERSION))
MODEL_PATH = os.path.join(EXPERIMENT_PATH, "checkpoints")
LOG_PATH = os.path.join(EXPERIMENT_PATH, "logs")
SNAPSHOT_PERIOD = 10000  # periodicity of saving current model

config = {
    "batch_size": 100,  # size of minibatch
    "experiment_path": EXPERIMENT_PATH,
    "latent_dim": 32,
    "log_period": 1000,
    "memory_size": 100000,
    "image_h": 84,
    "image_w": 84,
    "max_frames":50000
}


class PythonAIController(object):

    # Called at the started of the game
    def begin_play(self):
        logging.info("Begin Play on PythonAIController class")

        self.step_count = 0
        self.trainer = VAETrainer(config)
        self.trainer.init_training()
        self.trainer.load_model(MODEL_PATH)
        self.image_h, self.image_w = config["image_h"], config["image_w"]
        self.max_frames = config["max_frames"]


    def get_screen(self, screenshot):
        screenshot = np.frombuffer(screenshot, dtype=np.uint8)
        return screenshot.reshape((self.image_h, self.image_w, 1), order='F').swapaxes(0, 1)

    def tick(self,screenshot):
        start_time = time.clock()

        screen = self.get_screen(screenshot)

        # Skip frames when no screen is available.
        if screen is None or len(screen) == 0:
            return

        if self.step_count < self.max_frames:
            state = self.trainer.process_frame(screen)

            self.step_count += 1

            if self.step_count % SNAPSHOT_PERIOD == 0:
               self.trainer.save_model(MODEL_PATH)

            # Log elapsed time.
            finish_time = time.clock()
            elapsed = finish_time - start_time

        else:
            state = self.trainer.get_state(screenshot)

        # Log elapsed time.
        finish_time = time.clock()
        elapsed = finish_time - start_time

        return state