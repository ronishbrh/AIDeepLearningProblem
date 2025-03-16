#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

void print_vec_result(std::vector<float> &v) {
  for (float e : v)
    //std::cout << ((e > 0.5)? 1 : 0) << " ";
    std::cout << e << " ";
  std::cout << std::endl;
}

void print_image(std::vector<float> &v) {
  for (int i = 0; i<28; i++){
  	for (int j = 0; j<28; j++)
    	std::cout << ((v[i*28 + j] > 0.5f)? '.':'#') << "\t";
    std::cout << '\n';
  }
  std::cout << std::endl;
}

bool read_images_from_emnist(const std::string &filename, std::vector<std::vector<float>> &images){
	using namespace std;
	ifstream file(filename);
	if(file.is_open()){
		int magic_number;
		file.read(reinterpret_cast<char*>(&magic_number), 4);
		magic_number = __builtin_bswap32(magic_number);

		//cout << magic_number << endl;
		assert(magic_number == 0x803);

		int number_of_data;
		file.read(reinterpret_cast<char*>(&number_of_data), 4);
		number_of_data = __builtin_bswap32(number_of_data);
		cout << number_of_data << endl;

		int height;
		file.read(reinterpret_cast<char*>(&height), 4);
		height = __builtin_bswap32(height);
		cout << height << endl;
		assert(height == 28);

		int width;
		file.read(reinterpret_cast<char*>(&width), 4);
		width = __builtin_bswap32(width);
		cout << width << endl;
		assert(width == 28);

		for(int i = 0; i<number_of_data; i++){
			images.push_back(vector<float>(0));
			for(int j = 0; j<width*height; j++){
				unsigned char pixel;
				file.read(reinterpret_cast<char*>(&pixel), 1);
				images[i].push_back(pixel / 255.0f);
			}
		}

		file.close();
		return true;
	} else {
		cout << "Couldn't open the file." << endl;
		return false;
	}

}

bool read_labels_from_emnist(const std::string &filename, std::vector<std::vector<float>> &labels){
	using namespace std;
	ifstream file(filename);
	if(file.is_open()){
		int magic_number;
		file.read(reinterpret_cast<char*>(&magic_number), 4);
		magic_number = __builtin_bswap32(magic_number);

		assert(magic_number == 0x801);

		int number_of_data;
		file.read(reinterpret_cast<char*>(&number_of_data), 4);
		number_of_data = __builtin_bswap32(number_of_data);

		for(int i = 0; i<number_of_data; i++){
			unsigned char num;
			file.read(reinterpret_cast<char*>(&num), 1);
			labels.push_back({0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f});
			labels[i][num] = 1.0f;
		}

		file.close();
		return true;
	} else {
		cout << "Couldn't open the file." << endl;
		return false;
	}

}


int main(){
	using namespace std;

	vector<vector<float>> labels;
	if(!read_images_from_emnist("dataset/gzip/emnist-digits-train-images-idx3-ubyte", labels)){
		cerr << "Error occured" << endl;
		return 1;
	}

	print_image(labels[1]);
}
