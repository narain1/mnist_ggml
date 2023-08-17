#include <iostream>
#include <fstream>

#include "ggml.h"

using namespace std;

struct mnist_hparams {
  int32_t inp_channels = 1;
  int32_t out_channels = 8;
  int32_t kernel_size = 3;
  int32_t stride = 2;
  int32_t lin_inp = 288;
  int32_t n_classes = 10;
};

struct cnn_model {
  mnist_hparams hparams;
  struct ggml_tensor *conv1_weight;
  struct ggml_tensor *conv1_bias;
  struct ggml_tensor *bn1_bias;
  struct ggml_tensor *bn1_mean;
  struct ggml_tensor *lin1_weight;
  struct ggml_tensor *lin1_bias;

  struct ggml_context *ctx;
};

bool mnist_load_model(const std::string &fname, cnn_model &model) {
  fprintf(stderr, "%s: loading model from '%s' wait \n", __func__, fname.c_str());
  
  auto fin = std::ifstream(fname, ios::binary);
  if (!fin) {
    fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
    return false;
  }

  {
    uint32_t magic;
    fin.read((char *) &magic, sizeof(magic));
    if (magic != 0x67676d6c) {
      fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
      return false;
    }
  }

  auto &ctx = model.ctx;
  size_t ctx_size = 0;

  {
    const auto &hparams = model.hparams;

    const int32_t inp_channels = hparams.inp_channels;
    const int32_t out_channels = hparams.out_channels;
    const int32_t kernel_size = hparams.kernel_size;
    const int32_t stride = hparams.stride;
    const int32_t lin_inp = hparams.lin_inp;
    const int32_t n_classes = hparams.n_classes;

    ctx_size +=  inp_channels * out_channels * kernel_size * kernel_size * ggml_type_sizef(GGML_TYPE_F32);
    // conv bias
    ctx_size += out_channels * ggml_type_sizef(GGML_TYPE_F32);
    // bn weight, bias, mean, var
    ctx_size += 4 * out_channels * ggml_type_sizef(GGML_TYPE_F32);
    // linear weight and bias
    ctx_size += lin_inp * n_classes * ggml_type_sizef(GGML_TYPE_F32);
    ctx_size +=  n_classes * ggml_type_sizef(GGML_TYPE_F32);

    printf("%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size/(1024.0 * 1024.0));
  }

  {
    struct ggml_init_params params = {
      ctx_size + 1024 * 1024,
      NULL,
      false
    };

    model.ctx = ggml_init(params);
    if (!model.ctx) {
      fprintf(stderr, "%s: ggml_init() failed\n", __func__);
      return false;
    }
  }

  // read first conv layer
  {
    int32_t n_dims;
    const auto &hparams = model.hparams;

    fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
    {
      int32_t ne_weights[3] = {1, 1, 1};
      for (size_t i = 0; i < n_dims; i++) {
        fin.read(reinterpret_cast<char *> (&ne_weights[i]), sizeof(ne_weights[i]));
      }
      model.conv1_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 
          model.hparams.out_channels, 
          model.hparams.kernel_size, 
          model.hparams.kernel_size);
      fin.read(reinterpret_cast<char *>(model.conv1_weight->data), ggml_nbytes(model.conv1_weight));
      ggml_set_name(model.conv1_weight, "conv1_weight");
    }

    {
      fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
      int32_t ne_bias[n_dims];
      fin.read(reinterpret_cast<char *>(&ne_bias[0]), sizeof(ne_bias[0]));
      model.conv1_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model.hparams.out_channels);
      fin.read(reinterpret_cast<char *>(model.conv1_bias->data), ggml_nbytes(model.conv1_bias));
      ggml_set_name(model.conv1_bias, "conv1_bias");
    }
  }
  cout << "loaded convolution layer" << endl;
  // reading batchnorm layer 
  
  {
    int32_t n_dims;
    {
      fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
      int32_t ne_weight[n_dims];
      fin.read(reinterpret_cast<char *>(&ne_weight[0]), sizeof(ne_weight[0]));
      model.bn1_mean = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model.hparams.out_channels);
      fin.read(reinterpret_cast<char *>(model.bn1_mean->data), ggml_nbytes(model.bn1_mean));
      ggml_set_name(model.bn1_mean, "bn1_mean");
    }
    {
      fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
      int32_t ne_bias[n_dims];
      fin.read(reinterpret_cast<char *>(&ne_bias[0]), sizeof(ne_bias[0]));
      model.bn1_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model.hparams.out_channels);
      fin.read(reinterpret_cast<char *>(model.bn1_bias->data), ggml_nbytes(model.bn1_bias));
      ggml_set_name(model.bn1_bias, "bn1_bias");
    }
  }

  cout << "loaded batchnorm layer" << endl;

  // reading linear layer
  {
    int32_t n_dims;
    {
      fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
      int32_t ne_weight[n_dims];
      for (size_t i=0; i<n_dims; i++) {
        fin.read(reinterpret_cast<char *>(&ne_weight[i]), sizeof(ne_weight[i]));
      }
      model.lin1_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, model.hparams.lin_inp, model.hparams.n_classes);
      fin.read(reinterpret_cast<char *>(model.lin1_weight->data), ggml_nbytes(model.lin1_weight));
      ggml_set_name(model.lin1_weight, "lin1_weight");
    }

    {
      fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
      int32_t ne_bias[n_dims];
      fin.read(reinterpret_cast<char *>(&ne_bias[0]), sizeof(ne_bias[0]));
      model.lin1_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model.hparams.n_classes);
      fin.read(reinterpret_cast<char *>(model.lin1_bias->data), ggml_nbytes(model.lin1_bias));
      ggml_set_name(model.lin1_bias, "lin1_bias");
    }
  }

  cout << "loaded linear layer" << endl;

  // ggml_get_data(model.conv1_weight);
  for (size_t i = 0; i < 3; i++) { 
    for (size_t j = 0; j < 3; j++)
      cout << *(float *)((char *)model.conv1_weight -> data + (i * 3 + j) * model.conv1_weight->nb[1]) << " ";
    cout << endl;
  }
  fin.close();
  return true;
}

int main() {
  uint8_t buf[784];
  cnn_model model;

  mnist_load_model("models/ggml_model.bin", model);
  cout << "model successfully loaded" << endl;
  return 1;
}
