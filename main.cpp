#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include <ctime>
#include <cstdio>
#include <cmath>
#include "ggml.h"

using namespace std;

struct mnist_hparams {
  int32_t n_input = 784;
  int32_t n_hidden = 256;
  int32_t n_classes = 10;
};

struct mnist_model {
  mnist_hparams hparams;

  struct ggml_tensor *fc1_weight;
  struct ggml_tensor *fc1_bias;
  
  struct ggml_tensor *fc2_weight;
  struct ggml_tensor *fc2_bias;

  struct ggml_context *ctx;
};

bool mnist_model_load(const std::string &fname, mnist_model &model) {
  printf("%s: loading model from '%s'\n", __func__, fname.c_str());

  auto fin = std::ifstream(fname, ios::binary);
  if (!fin) {
    fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
    return false;
  }

  {
    uint32_t magic;
    fin.read((char *) &magic, sizeof(magic));
    if (magic != GGML_FILE_MAGIC) {
      fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
      return false;
    }
  }

  auto &ctx = model.ctx;
  size_t ctx_size = 0;

  {
    const auto &hparams = model.hparams;
    const int n_input = hparams.n_input;
    const int n_hidden = hparams.n_hidden;
    const int n_classes = hparams.n_classes;

    ctx_size += n_input * n_hidden * ggml_type_sizef(GGML_TYPE_F32);
    ctx_size +=           n_hidden * ggml_type_sizef(GGML_TYPE_F32);

    ctx_size += n_hidden * n_classes * ggml_type_sizef(GGML_TYPE_F32);
    ctx_size +=            n_classes * ggml_type_sizef(GGML_TYPE_F32);

    printf("%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size/(1024.0 * 1024.0));
  }

  {
    struct ggml_init_params params = {
      ctx_size + 1024 * 1024,
      NULL,
      false,
    };

    model.ctx = ggml_init(params);
    if (!model.ctx) {
      fprintf(stderr, "%s: ggml_init() failed\n", __func__);
      return false;
    }
  }

  // read fc1 layer
  {
    int32_t n_dims;
    fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
    {
      // reading dimensions of tensor
      int32_t ne_weight[2] = {1, 1};
      for (int i = 0; i < n_dims; ++i) {
        fin.read(reinterpret_cast<char *>(&ne_weight[i]), sizeof(ne_weight[i]));
      }

      model.hparams.n_input = ne_weight[0];
      model.hparams.n_hidden = ne_weight[1];

      model.fc1_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, model.hparams.n_input, model.hparams.n_hidden);
      fin.read(reinterpret_cast<char *>(model.fc1_weight->data), ggml_nbytes(model.fc1_weight));
      ggml_set_name(model.fc1_weight, "fc1_weight");
    }

    {
      int32_t ne_bias[2] = {1, 1};
      for (int i = 0; i < n_dims; ++i) {
        fin.read(reinterpret_cast<char *>(&ne_bias[i]), sizeof(ne_bias[i]));
      }

      model.fc1_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model.hparams.n_hidden);
      fin.read(reinterpret_cast<char *>(model.fc1_bias->data), ggml_nbytes(model.fc1_bias));
      ggml_set_name(model.fc1_bias, "fc1_bias");

      //model.fc1_bias->op_params[0] = 0xdeadbeef;
    }
  }

  {
    int32_t n_dims;
    fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));

    {
      int32_t ne_weight[2] = {1, 1};
      for (int i=0; i < n_dims; ++i) {
        fin.read(reinterpret_cast<char *>(&ne_weight[i]), sizeof(ne_weight[i]));
      }

      model.fc2_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, model.hparams.n_hidden, model.hparams.n_classes);
      fin.read(reinterpret_cast<char *>(model.fc2_weight->data), ggml_nbytes(model.fc2_weight));
      ggml_set_name(model.fc2_weight, "fc2_weight");
    }

    {
      int32_t ne_bias[2] = {1, 1};
      for (int i = 0; i < n_dims; ++i) {
        fin.read(reinterpret_cast<char *>(&ne_bias[i]), sizeof(ne_bias[i]));
      }

      model.fc2_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model.hparams.n_classes);
      fin.read(reinterpret_cast<char *>(model.fc2_bias->data), ggml_nbytes(model.fc2_bias));
      ggml_set_name(model.fc2_bias, "fc2_bias");
    }
  }

  fin.close();

  return true;
}

int mnist_eval(
    const mnist_model &model,
    const int n_threads,
    std::vector<float> digit,
    const char *fname_cgraph
    ) {
  
  const auto &hparams = model.hparams;

  static size_t buf_size = hparams.n_input * sizeof(float) * 4;
  static void *buf = malloc(buf_size);

  struct ggml_init_params params = {
    buf_size,
    buf,
    false
  };

  struct ggml_context *ctx0 = ggml_init(params);
  struct ggml_cgraph gf = {};

  struct ggml_tensor *input = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hparams.n_input);
  memcpy(input->data, digit.data(), ggml_nbytes(input));
  ggml_set_name(input, "input");

  ggml_tensor *fc1 = ggml_add(ctx0, ggml_mul_mat(ctx0, model.fc1_weight, input), model.fc1_bias);
  ggml_tensor *fc2 = ggml_add(ctx0, ggml_mul_mat(ctx0, model.fc2_weight, ggml_relu(ctx0, fc1)), model.fc2_bias);

  ggml_tensor *probs = ggml_soft_max(ctx0, fc2);
  ggml_set_name(probs, "probs");

  ggml_build_forward_expand(&gf, probs);
  ggml_graph_compute_with_ctx(ctx0, &gf, n_threads);


  const float *probs_data = ggml_get_data_f32(probs);

  const int prediction = max_element(probs_data, probs_data + 10) - probs_data;
  ggml_free(ctx0);

  return prediction;
}


int main(int argc, char **argv) {
  srand(time(NULL));
  ggml_time_init();
  
  if (argc != 3) {
    fprintf(stderr, "Usage: %s models/mnist/ggml-model-f32.bin models/mnist/t10k-images.idx3-ubyte\n", argv[0]);
    exit(0);
  }

  cout << "here";
  uint8_t buf[784];
  mnist_model model;
  int prediction;

  // load the model
  
  {
    const int64_t t_start_us = ggml_time_us();
    if (!mnist_model_load(argv[1], model)) {
      fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, "models/ggml-model-f32.bin");
      return 1;
    }
    
    cout << "successfully loaded the model" << endl;

  }

  {
    auto fin = ifstream("models/t10k-images.idx3-ubyte", ios::binary);
    if (!fin) {
      fprintf(stderr, "failed to open digits file");
      return 0;
    }

    for (size_t i = 0; i < 10000; i++) {
          fin.seekg(16 + 784 * i);
          fin.read((char *) &buf, sizeof(buf));
          vector<float> digit(begin(buf), end(buf));
          for(int i=0; i < digit.size(); ++i)
            digit[i] = digit[i]/255.0f;
          prediction = mnist_eval(model, 1, digit, "mnist.ggml"); 
      }
  }

  for (size_t row = 0; row < 28; row++) {
    for (size_t col = 0; col < 28; col++)
      fprintf(stderr, "%c ", (float)buf[row * 28 + col] > 210 ? '*' : '-');
    fprintf(stderr, "\n");
  }
  cout << endl;
  cout << prediction << endl;
  return 0;
}
