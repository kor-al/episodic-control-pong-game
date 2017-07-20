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
        self.vae = VAE(self.latent_dim, self.batch_size)
        # Tools for saving and loading networks
        self.saver = tf.train.Saver()

        self.step = tf.Variable(0, name="step")
        self.increment_step = self.step.assign_add(1)

        self.newscore = tf.placeholder(tf.int32)
        self.score = tf.Variable(0, name="score")
        self.assign_score = tf.assign(self.score, self.newscore);
        self.summariesScore = tf.summary.scalar("score", tf.cast(self.score, tf.int32))
        #self.summariesScore = tf.summary.merge_all();

    def init_training(self):
        # Initialize training parameters
        self.session.run(tf.global_variables_initializer())
        self.t = self.step.eval(self.session) #frames count


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
        self.saver.save(self.session, path + "/vae", global_step = self.t)

    def make_train_step(self, obs):
        # sample a minibatch to train on
        minibatch = (np.array(self.memory.sample(self.batch_size))>0.5).astype(np.float32)
        return self.vae.train(self.session, minibatch)

    def process_frame(self, screen):

        # store the transition in memory
        self.memory.add_experience(screen)

        summaries, elbo, z = self.make_train_step(screen)
        self.memory.current_elbo = elbo

        # update the old values
        self.t = self.session.run(self.increment_step)

        # print info
        if self.t % self.log_period == 0:
            now_string = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
            message = "TIME {}, FRAMES {}, ELBO {}\n".format(now_string, self.t, self.memory.current_elbo)

            self.summary_output.write(message)
            self.summary_output.flush()


            if summaries is not None:
                self.summary_writer.add_summary(summaries, self.t)

        return z

    def get_state(self, screen, score):

        _ , score_summary, self.t =self.session.run([self.assign_score, self.summariesScore, self.increment_step], feed_dict={self.newscore: score})

        if self.t % self.log_period == 0:
            if score_summary is not None:
                self.summary_writer.add_summary(score_summary, self.t)

        return self.vae.get_state(self.session, screen)