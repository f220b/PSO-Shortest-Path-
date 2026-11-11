import math
import obstacles as ob
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import time

DIM = 13
START = ob.Point(1, 1)
END = ob.Point(10, 20)


class Particle:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.best_position = position
        self.best_loss = float('inf')


def calculate_path_length(points):
    length = 0
    for i in range(1, len(points)):
        length += math.sqrt((points[i][0] - points[i - 1][0]) ** 2 + (points[i][1] - points[i - 1][1]) ** 2)
    return length


def plot_circle(circle):
    plt.gca().add_patch(
        plt.Circle(
            (circle.C.x, circle.C.y), circle.r - 0.2, edgecolor="red", facecolor="red"
        )
    )


def plot_graph(best_location, path_length, execution_time, obstacles):
    plt.close("all")
    plt.figure(figsize=(7, 7))
    plt.grid("on")
    plt.rc("axes", axisbelow=True)
    plt.scatter(END.x, END.y, 100, marker="*", facecolors="k", edgecolors="k")
    plt.scatter(START.x, START.y, 100, marker="o", facecolors="k", edgecolors="k")
    X = [best_location[i][0] for i in range(DIM)]
    X.insert(0, START.x)
    X.append(END.x)
    Y = [best_location[i][1] for i in range(DIM)]
    Y.insert(0, START.y)
    Y.append(END.y)
    plt.plot(X, Y, "o-")
    for i in range(DIM):
        plt.scatter(
            best_location[i][0],
            best_location[i][1],
            25,
            marker=".",
            facecolors="blue",
            edgecolors="face",
        )
    for obstacle in obstacles:
        plot_circle(obstacle)

    plt.xlim(-1, 23)
    plt.ylim(-1, 23)
    plt.title("Particle Swarm Optimization")

    plt.text(
        22,
        22,
        f"Path Length: {path_length:.2f}\nExecution Time: {execution_time:.2f} seconds",
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.show()


def random_initialization(swarm_size):
    particles = []
    for _ in range(swarm_size):
        position = np.random.rand(DIM, 2) * 20
        velocity = np.random.rand(DIM, 2)
        particle = Particle(position, velocity)
        particles.append(particle)

    return particles


def loss_function(x):
    z = (x[0, 0] - START.x) ** 2 + (x[0, 1] - START.y) ** 2
    for i in range(DIM - 1):
        z = z + ((x[i, 0] - x[i + 1, 0]) ** 2 + (x[i, 1] - x[i + 1, 1]) ** 2)
    z = z + (x[DIM - 1, 0] - END.x) ** 2 + (x[DIM - 1, 1] - END.y) ** 2
    return math.sqrt(z)


def is_valid(circles, p):
    to_add = ob.Point(0, 0)
    point_p = ob.Point(p[0], p[1])
    for i in range(len(circles)):
        if circles[i].inside_circle(point_p):
            to_add = ob.Point(
                circles[i].how_to_exit_x(point_p.x) + to_add.x,
                circles[i].how_to_exit_y(point_p.y) + to_add.y,
            )
    return to_add


def particle_swarm_optimization(max_iterations, swarm_size, max_vel, step_size, inertia, c1, c2, circles):
    particles = random_initialization(swarm_size)
    global_best_location = np.random.rand(DIM, 2) * 20
    global_best_loss = float('inf')
    best_location = []

    for iteration_i in range(max_iterations):
        for particle in particles:
            dim_i = 0
            while dim_i < DIM:
                error_particle_best = particle.best_position[dim_i] - particle.position[dim_i]
                error_global_best = global_best_location[dim_i] - particle.position[dim_i]
                new_vel = (inertia * min(1, (dim_i ** (0.5)) / 4) * particle.velocity[dim_i] +
                           c1 * np.random.rand(1) * error_particle_best +
                           c2 * np.random.rand(1) * error_global_best)

                if new_vel[0] < -max_vel:
                    new_vel[0] = -max_vel
                elif new_vel[0] > max_vel:
                    new_vel[0] = max_vel
                if new_vel[1] < -max_vel:
                    new_vel[1] = -max_vel
                elif new_vel[1] > max_vel:
                    new_vel[1] = max_vel

                particle.position[dim_i] = particle.position[dim_i] + new_vel[:] * step_size
                particle.velocity[dim_i] = new_vel[:]

                particle_help = is_valid(circles, particle.position[dim_i, :])
                particle.position[dim_i, 0] += particle_help.x
                particle.position[dim_i, 1] += particle_help.y
                if abs(particle_help.x) > 0.1 or abs(particle_help.y) > 0.1:
                    dim_i -= 1
                dim_i += 1

            particle_error = loss_function(particle.position)
            if particle_error < particle.best_loss:
                particle.best_loss = particle_error
                particle.best_position = particle.position.copy()

            if particle_error < global_best_loss:
                global_best_loss = particle_error
                global_best_location = particle.position.copy()

        best_location = global_best_location.copy()

    return best_location


def main():
    print("Please wait:")
    obstacle1 = ob.Obstacle_Circle(2, ob.Point(9, 14))
    obstacle2 = ob.Obstacle_Circle(3.5, ob.Point(5, 6))
    obstacle3 = ob.Obstacle_Circle(2, ob.Point(11.5, 9))
    obstacles = [obstacle1, obstacle2, obstacle3]

    start_time = time.time()
    best_location = particle_swarm_optimization(
        max_iterations=1000,
        swarm_size=100,
        max_vel=3,
        step_size=1,
        inertia=0.9,
        c1=2.05,
        c2=2.05,
        circles=obstacles,
    )
    end_time = time.time()
    execution_time = end_time - start_time

    path_length = calculate_path_length(best_location)
    print("100% completed!")
    plot_graph(best_location, path_length, execution_time, obstacles)


if __name__ == "__main__":
    main()
