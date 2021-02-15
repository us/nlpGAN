import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import os
from models.generator import Generator
from models.discriminator import Discriminator
from models.rollout import ROLLOUT
from settings import *


def generator_dataloader(data_file, batch_size):
    token_stream = []
    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split()
            parse_line = [int(x) for x in line]

            if len(parse_line) == 20:
                token_stream.append(parse_line)

    return tf.data.Dataset.from_tensor_slices(token_stream).shuffle(len(token_stream)).batch(batch_size)


def discriminator_dataloader(positive_file, negative_file, batch_size):
    examples = []
    labels = []
    with open(positive_file) as fin:
        for line in fin:
            line = line.strip()
            line = line.split()
            parse_line = [int(x) for x in line]
            if len(parse_line) == 20:
                examples.append(parse_line)
                labels.append([0, 1])

    with open(negative_file) as fin:
        for line in fin:
            line = line.strip()
            line = line.split()
            parse_line = [int(x) for x in line]
            if len(parse_line) == 20:
                examples.append(parse_line)
                labels.append([1, 0])
    return tf.data.Dataset.from_tensor_slices((examples, labels)).shuffle(len(examples)).batch(batch_size)


if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        for dev in physical_devices:
            tf.config.experimental.set_memory_growth(dev, True)

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH)
    discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size,
                                  embedding_size=dis_embedding_dim,
                                  filter_sizes=dis_filter_sizes, num_filters=dis_num_filters,
                                  dropout_keep_prob=dis_dropout_keep_prob,
                                  l2_reg_lambda=dis_l2_reg_lambda)
    print("== generator model summary ==")
    generator.generator_model.summary()
    print("== discriminator model summary ==")
    discriminator.d_model.summary()
    gen_dataset = generator_dataloader(positive_file, BATCH_SIZE)

    if not os.path.exists("pretrained_models"):
        os.makedirs("pretrained_models")

    if not os.path.exists(pretrained_generator_file):
        print('Start pre-training generator')
        gen_history = generator.pretrain(gen_dataset, PRE_EPOCH_NUM, generated_num // BATCH_SIZE)
        generator.save(pretrained_generator_file)
        print('Finished pre-training generator...')
    else:
        generator.load(pretrained_generator_file)
        gen_history = generator.pretrain(gen_dataset, 1, generated_num // BATCH_SIZE)

    if not os.path.exists(pretrained_discriminator_file):
        print('Start pre-training discriminator...')
        for _ in range(50):
            generator.generate_samples(generated_num // BATCH_SIZE, negative_file)
            dis_dataset = discriminator_dataloader(positive_file, negative_file, BATCH_SIZE)
            discriminator.train(dis_dataset, 3, (generated_num // BATCH_SIZE) * 2)
        discriminator.save(pretrained_discriminator_file)
        print('Finished pre-training discriminator...')
    else:
        discriminator.load(pretrained_discriminator_file)
    rollout = ROLLOUT(generator, 0.8)

    discriminator.loss_history = []
    discriminator.acc_history = []
    print('Start Training...')
    for epoch in range(EPOCH_NUM):
        print("Generator", epoch)
        for it in range(1):
            samples = generator.generate_one_batch()
            rewards = rollout.get_reward(samples, 16, discriminator)
            generator.train_step(samples, rewards)

        rollout.update_params()

        print("Discriminator", epoch)
        for _ in range(5):
            generator.generate_samples(generated_num // BATCH_SIZE, negative_file)
            dis_dataset = discriminator_dataloader(positive_file, negative_file, BATCH_SIZE)
            discriminator.train(dis_dataset, 3, (generated_num // BATCH_SIZE) * 2)
    generator.save(generator_file)
    discriminator.save(discriminator_file)

    # print(history.history.keys()) # dict_keys(['loss'])
    # generator loss
    plt.plot(gen_history.history['loss'])
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    # plt.show()
    plt.savefig('generator_loss.png')
    plt.close()
    # discriminator accuracy
    # print(disc_history.history.keys())
    x = np.array(range(len(discriminator.acc_history))) / (5*3)
    y = discriminator.acc_history
    plt.plot(x,y)
    plt.title('discriminator accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    # plt.show()
    plt.savefig('discriminator_accuracy.png')
    plt.close()

    # discriminator loss
    x = np.array(range(len(discriminator.loss_history))) / (5*3)
    y = discriminator.loss_history
    plt.plot(discriminator.loss_history)
    plt.title('discriminator loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    # plt.show()
    plt.savefig('discriminator_loss.png')
    plt.close()

    generator.generate_samples(generated_num // BATCH_SIZE, generated_file)
