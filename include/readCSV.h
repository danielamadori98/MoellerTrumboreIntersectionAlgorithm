#ifndef CSV_READER_H
#define CSV_READER_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

double** read_CSV_double(const std::string& filename, unsigned short columnsSize, char separator, unsigned short& rowsCounter) {
	std::ifstream file(filename);

	if (!file.is_open()) {
		std::cerr << "Error: Unable to open file " << filename << std::endl;
		return nullptr;
	}

	std::vector<std::vector<double>> tempMatrix;
	std::string line;

	while (std::getline(file, line)) {
		std::istringstream iss(line);
		std::string token;
		std::vector<double> row;

		while (std::getline(iss, token, separator)) {
			try {
				row.push_back(std::stod(token));
			}
			catch (const std::invalid_argument& e) {
				std::cerr << "Error: Invalid double in CSV file: " << token << std::endl;
				for (auto& r : tempMatrix)
					r.clear();
				
				return nullptr;
			}
		}

		if (row.size() != columnsSize) {
			std::cerr << "Error: Invalid number of columns in CSV file, expected " << columnsSize << std::endl;
			for (auto& r : tempMatrix)
				r.clear();
			
			return nullptr;
		}

		tempMatrix.push_back(row);
	}

	file.close();

	rowsCounter = tempMatrix.size();

	// Convert vector of vectors to a 2D array
	double** dataset = new double* [rowsCounter];
	for (size_t i = 0; i < rowsCounter; ++i) {
		dataset[i] = new double[columnsSize];
		std::copy(tempMatrix[i].begin(), tempMatrix[i].end(), dataset[i]);
	}

	return dataset;
}


unsigned short** read_CSV_unsigned_short(const std::string& filename, unsigned short columnsSize, char separator, unsigned short& rowsCounter) {
	std::ifstream file(filename);

	if (!file.is_open()) {
		std::cerr << "Error: Unable to open file " << filename << std::endl;
		return nullptr;
	}

	std::vector<std::vector<unsigned short>> tempMatrix;
	std::string line;

	while (std::getline(file, line)) {
		std::istringstream iss(line);
		std::string token;
		std::vector<unsigned short> row;

		while (std::getline(iss, token, separator)) {
			try {
				row.push_back(static_cast<unsigned short>(std::stoul(token)));
			}
			catch (const std::invalid_argument& e) {
				std::cerr << "Error: Invalid unsigned short in CSV file: " << token << std::endl;
				for (auto& r : tempMatrix) 
					r.clear();
				
				return nullptr;
			}
		}

		if (row.size() != columnsSize) {
			std::cerr << "Error: Invalid number of columns in CSV file, expected " << columnsSize << std::endl;
			for (auto& r : tempMatrix)
				r.clear();
			
			return nullptr;
		}

		tempMatrix.push_back(row);
	}

	file.close();

	rowsCounter = tempMatrix.size();

	// Convert vector of vectors to a 2D array
	unsigned short** dataset = new unsigned short* [rowsCounter];
	for (size_t i = 0; i < rowsCounter; ++i) {
		dataset[i] = new unsigned short[columnsSize];
		std::copy(tempMatrix[i].begin(), tempMatrix[i].end(), dataset[i]);
	}

	return dataset;
}

bool** read_CSV_bool(const std::string& filename, unsigned short columnsSize, char separator, unsigned short& rowsCounter) {
	std::ifstream file(filename);

	if (!file.is_open()) {
		std::cerr << "Error: Unable to open file " << filename << std::endl;
		return nullptr;
	}

	std::vector<std::vector<bool>> tempMatrix;
	std::string line;

	while (std::getline(file, line)) {
		std::istringstream iss(line);
		std::string token;
		std::vector<bool> row;

		while (std::getline(iss, token, separator)) {
			try {
				row.push_back(static_cast<bool>(std::stoi(token)));
			}
			catch (const std::invalid_argument& e) {
				std::cerr << "Error: Invalid boolean in CSV file: " << token << std::endl;
				for (auto& r : tempMatrix)
					r.clear();
				
				return nullptr;
			}
		}

		if (row.size() != columnsSize) {
			std::cerr << "Error: Invalid number of columns in CSV file, expected " << columnsSize << std::endl;
			for (auto& r : tempMatrix)
				r.clear();
			
			return nullptr;
		}

		tempMatrix.push_back(row);
	}

	file.close();

	rowsCounter = tempMatrix.size();

	// Convert vector of vectors to a 2D array
	bool** dataset = new bool* [rowsCounter];
	for (size_t i = 0; i < rowsCounter; ++i) {
		dataset[i] = new bool[columnsSize];
		std::copy(tempMatrix[i].begin(), tempMatrix[i].end(), dataset[i]);
	}

	return dataset;
}

#endif