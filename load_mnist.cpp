#include <fstream>
#include <iostream>

using namespace std;

int main() {
  uint8_t buf[784];
  auto fin = std::ifstream("models/t10k-images.idx3-ubyte", ios::binary);
  if (!fin) {
    fprintf(stderr, "failed to open digits file\n");
    return 0;
  }

  srand(time(NULL));

  uint32_t targ = rand() % 10000;
  uint32_t digit[784];
  fin.seekg(16 + 784 * targ);
  fin.read((char *) &buf, sizeof(buf));

  {
    for (int row = 0; row < 28; row++) {
      for (int col = 0; col < 28; col++)
        fprintf(stderr, "%c ", (float)buf[row*28 +col] > 230 ? '*' : '-');
      fprintf(stderr, "\n");
    } 
    fprintf(stderr, "\n");
  }
  cout << targ << endl; 
  return 0;
}
