#!/bin/bash -eux

DATA_DIR="data"

TRAIN_DATASET_ID="1FU0LeAddXldec07HPb-8Ukj0ZNUe2jcQ"
TEST_DATASET_ID="1gdyc39FbGumZv8KXyE8CTyKGOczn1DAS"

TRAIN_DATA_FILE="fashion-mnist_train.zip"
TEST_DATA_FILE="fashion-mnist_test.zip"

ROOT_URL="https://docs.google.com/uc?export=download&id="

wget --no-check-certificate -r "${ROOT_URL}${TRAIN_DATASET_ID}" -O "${DATA_DIR}/${TRAIN_DATA_FILE}"
wget --no-check-certificate -r "${ROOT_URL}${TEST_DATASET_ID}" -O "${DATA_DIR}/${TEST_DATA_FILE}"

unzip "${DATA_DIR}/${TRAIN_DATA_FILE}" -d ${DATA_DIR}
unzip "${DATA_DIR}/${TEST_DATA_FILE}" -d ${DATA_DIR}

rm "${DATA_DIR}/${TRAIN_DATA_FILE}"
rm "${DATA_DIR}/${TEST_DATA_FILE}"
