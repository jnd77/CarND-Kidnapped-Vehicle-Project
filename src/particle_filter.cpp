/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */

  if (is_initialized)
    return;

  num_particles = 1000;

  // The random number generator
  std::default_random_engine gen;

  // The 3 Gaussian distributions for x, y and theta
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  // Initialize the set of particles
  for(int i = 0; i < num_particles; i++) {
    Particle p;
	p.id = i;
	p.x = dist_x(gen);
	p.y = dist_y(gen);
	p.theta = dist_theta(gen);
	p.weight = 1;

	// Add it to the list
	particles.push_back(p);
  }
  
  // All initialized
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  // The random number generator
  std::default_random_engine gen;
  // The 3 Gaussian distributions for x, y and theta
  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0, std_pos[2]);

  // Update each particle
  for(int i = 0; i < num_particles; ++i) {
	// To avoid divisions by zero, special case when yaw_rate is small
	float theta_zero = particles[i].theta;
	float theta_f = theta_zero + yaw_rate * delta_t;
	if (yaw_rate <= fabs(1e-6)) {
		particles[i].x += velocity * delta_t * cos(theta_zero);
		particles[i].y += velocity * delta_t * sin(theta_zero);
	} else {
		particles[i].x += velocity / yaw_rate * (sin(theta_f) - sin(theta_zero));
		particles[i].y += velocity / yaw_rate * (cos(theta_zero) - cos(theta_f));
	}
	particles[i].theta = theta_f;

	// Add noise
	particles[i].x += dist_x(gen);
	particles[i].y += dist_y(gen);
	particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  int num_landmarks = predicted.size();
  int num_obs = observations.size();
  for(int i = 0; i < num_landmarks; i++) {
	float x1 = predicted[i].x;
	float y1 = predicted[i].y;

    // Find the nearest observation
	float min_distance = 1e8;
	int nearest_obs = -1;
	for(int j = 0; j < num_obs; j++) {
	  float distance = dist(x1, y1, observations[j].x, observations[j].y);
	  if (distance < min_distance) {
	    min_distance = distance;
		nearest_obs = j;
	  }
	}

	// Set the association
	observations[nearest_obs].id = predicted[i].id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  int num_obs = observations.size();
  int num_landmarks = map_landmarks.landmark_list.size();

  // Loop over all particles
  for(int i = 0; i < num_particles; i++) {
    Particle p = particles[i];
    
	// First need to transform the observations coordinates from vehicle to map coordinates
    vector<LandmarkObs> obs_in_map_coords;
	for(int j = 0; j < num_obs; j++) {
	  LandmarkObs obs;
	  obs.x = p.x + cos(p.theta) * observations[j].x - sin(p.theta) * observations[j].y;
	  obs.y = p.y + sin(p.theta) * observations[j].x + cos(p.theta) * observations[j].y;
	  obs_in_map_coords.push_back(obs);
	}

	// Second keep the landmarks within sensor range
    vector<LandmarkObs> landmarks;
	for(int j = 0; j < num_landmarks; j++) {
	  float landmark_x = map_landmarks.landmark_list[j].x_f;
	  float landmark_y = map_landmarks.landmark_list[j].y_f;
	  if (dist(p.x, p.y, landmark_x, landmark_y) < sensor_range) {
		  LandmarkObs landmark;
		  landmark.x = landmark_x;
		  landmark.y = landmark_y;
		  landmark.id = map_landmarks.landmark_list[j].id_i;
		  landmarks.push_back(landmark);
	  }
	}

	// Then do the association
    dataAssociation(landmarks, obs_in_map_coords);

	// Finally compute the new weight
	int num_landmarks_within_range = landmarks.size();
	double weight = 1;
	for(int j = 0; j < num_landmarks_within_range; j++) {
		// Find the associated observation
		LandmarkObs associated_obs;
		for(int k = 0; k < num_obs; k++) {
		  if (obs_in_map_coords[k].id == landmarks[j].id) {
		  	  associated_obs = obs_in_map_coords[k];
			  break;
		  }
		}

		weight *= exp(-0.5 * (pow(associated_obs.x - landmarks[j].x, 2) / pow(std_landmark[0], 2) +
							  pow(associated_obs.y - landmarks[j].y, 2) / pow(std_landmark[1], 2))) /
				  (2 * M_PI * std_landmark[0] * std_landmark[1]);
	}

	// Update the weight
	particles[i].weight = weight;
  }

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  // The random number generator
  std::default_random_engine gen;

  // Set the discrete weights
  vector<double> weights;
  for(int i = 0; i < num_particles; i++)
    weights.push_back(particles[i].weight);

  std::discrete_distribution<> dist_particles(weights.begin(), weights.end());
  vector<Particle> sampled_particles;

  // Sample the particles
  for(int i = 0; i < num_particles; i++) {
    int sampled_particle = dist_particles(gen);
	sampled_particles.push_back(particles[sampled_particle]);
  }

  particles = sampled_particles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}