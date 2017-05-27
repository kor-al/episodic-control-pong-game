import tensorflow as tf
import numpy as np
import sys
import os
import datetime
from vae import VAE
from memory import ExperienceMemory

class VAETrainer(object):
    def __init__(self, config):
        # Create session to store trained parameters
        self.session = tf.Session()

        # Create and configure logging directories and file handles.
        experiment_path = config["experiment_path"]
        os.makedirs(experiment_path, exist_ok=True)
        summary_dir = os.path.join(experiment_path, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        self.summary_writer = tf.summary.FileWriter(summary_dir)
        self.summary_output = open(os.path.join(experiment_path, "log.txt"), "a")

        self.batch_size = config["batch_size"]
        self.latent_dim = config["latent_dim"]
        self.log_period = config["log_period"]
        self.max_frames = config["max_frames"]
        self.memory = ExperienceMemory(config["memory_size"])

        # Create agent for training
        self.vae=VAE(self.latent_dim)
        # Tools for saving and loading networks
        self.saver = tf.train.Saver()

        self.frames_count = 0

    def init_training(self):
        # Initialize training parameters
        self.session.run(tf.global_variables_initializer())


    def load_model(self, path):
        checkpoint = tf.train.get_checkpoint_state(path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded: {}".format(checkpoint.model_checkpoint_path))
        else:
            print("Could not find old network weights")

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.saver.save(self.session, path + "/vae")

    def make_train_step(self, obs):
        # sample a minibatch to train on
        minibatch = list(self.memory.sample(self.batch_size))
        return self.vae.train(self.session, minibatch)

    def process_frame(self, screen):

        # store the transition in memory
        self.memory.add_experience(screen)

        summaries, loss, z = self.make_train_step(screen)
        self.memory.current_elbo = loss

        # print info
        if self.frames_count % self.log_period == 0:
            now_string = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
            message = "TIME {}, FRAMES {}, ELBO {}\n".format(now_string, self.frames_count, self.memory.current_elbo)

            self.summary_output.write(message)
            self.summary_output.flush()

            if summaries is not None:
                self.summary_writer.add_summary(summaries)

        self.frames_count += 1

        return z

    def get_state(self, screen):
        return self.vae.get_state(self.session, screen)