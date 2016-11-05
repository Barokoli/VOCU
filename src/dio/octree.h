/*
 * octree.h
 *
 *  Created on: 03.11.2016
 *      Author: sebastian
 */

#ifndef OCTREE_H_
#define OCTREE_H_

class Octree{
	public:
	void init_from_random(int sqr_size);
	bool calculated;
private:
	void *data_pointer;
};

void Octree::init_from_random(int sqr_size){
	std::cout << "Initializing Octree from random values. Size: (" << sqr_size << "," << sqr_size << "," << sqr_size << ")\nOn Thread:" << std::this_thread::get_id() << std::endl;
	std::this_thread::sleep_for (std::chrono::seconds(5));
	std::cout << "init finished" << std::endl;
	calculated = true;
}

#endif /* OCTREE_H_ */
