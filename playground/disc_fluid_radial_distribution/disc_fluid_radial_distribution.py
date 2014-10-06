from __future__ import division
import numpy as np
from mcpele.monte_carlo import _BaseMCRunner, RandomCoordsDisplacement

class MC(_BaseMCRunner):
    def set_control(self, temp):
        self.set_temperature(temp)
        
class ComputeCompareGr(nr_particles=42, volume_fraction=0.314159265358979, nr_eq_steps=1e4, nr_record_steps=1e4, all_particle_moves=True):
    def __init__(self):
        self.diameter = 1.0
        self.box_dimension = 2
        self.nr_particles = nr_particles
        self.volume_fraction = volume_fraction
        self.density = 4 * self.volume_fraction / (np.pi * self.diameter**2)
        self.nr_eq_steps = nr_eq_steps
        self.nr_record_steps = nr_record_steps
        self.box_length = np.sqrt(self.nr_particles * np.pi * self.diameter**2 / (4 * self.volume_fraction))
        self.all_particle_moves = all_particle_moves
        self.run()
    def run(self):
        self.get_initial_configuration()
        self.set_up_mc()
        self.equilibrate()
        self.record_gr()
        self.print_pressure()
        self.plot_gr()
    def get_initial_configuration(self):
        boxx = self.box_length
        boxy = self.box_length
        maximum_radius = 0.5 * self.diameter
        minimum_spacing_x = 2 * maximum_radius
        minimum_spacing_y = np.sqrt(3) * 0.5 * minimum_spacing_x
        LX = int(boxx / minimum_spacing_x)
        LY = int(boxy / minimum_spacing_y)
        max_placable_discs = LX * LY
        if self.nr_particles > max_placable_discs:
            raise Exception("finding legal initial configuration failed")
        while ((LX - 1) * (LY - 1)) >= self.nr_particles:
            LX -= 1
            LY -= 1
        spacing_x = boxx / LX 
        spacing_y = boxy / LY
        self.coords = np.zeros(self.nr_particles * self.box_dimension)
        for i in xrange(self.nr_particles):
            xi = self.box_dimension * i
            xint = i % LX
            yint = int(i / LX)
            self.coords[xi] = (xint + 0.5 * (yint % 2)) * spacing_x
            self.coords[xi + 1] = yint * spacing_y
    def set_up_mc(self):
        self.temp = 1
        self.total_nr_steps = self.nr_eq_steps + self.nr_record_steps
        self.mc = MC(self.potential, self.coords, self.temp, self.total_nr_steps)
        # TODO: problem: there is only an overlap test in basinvolume
        
if __name__ == "__main__":
    ComputeCompareGr(nr_particles=42, volume_fraction=0.314159265358979, nr_eq_steps=1e4, nr_record_steps=1e4, all_particle_moves=True)