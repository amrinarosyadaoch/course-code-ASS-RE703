# course-code-ASS-RE703
import numpy as np
import matplotlib.pyplot as plt

# Fungsi fitness untuk mengevaluasi performa PID
def fitness_function(params, setpoint, actual_response):
    Kp, Ki, Kd = params
    error = setpoint - actual_response
    
    # Simulasi sederhana error steady-state, overshoot, dan rise time
    overshoot = max(0, max(actual_response) - setpoint)
    steady_state_error = abs(error[-1])
    rise_time = np.argmax(actual_response >= 0.9 * setpoint) if max(actual_response) >= 0.9 * setpoint else len(actual_response)

    # Fitness: kombinasi penalti untuk overshoot, steady-state error, dan rise time
    return overshoot + steady_state_error + rise_time / len(actual_response)

# Parameter PSO
def pso(optimize_func, bounds, num_particles=30, max_iter=100):
    dim = len(bounds)  # Dimensi (Kp, Ki, Kd)
    particles = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (num_particles, dim))
    velocities = np.zeros_like(particles)
    local_best_positions = particles.copy()
    global_best_position = particles[0]

    local_best_scores = np.array([float('inf')] * num_particles)
    global_best_score = float('inf')

    w, c1, c2 = 0.5, 1.5, 1.5  # PSO coefficients

    for iteration in range(max_iter):
        for i, particle in enumerate(particles):
            response = simulate_system(particle)  # Fungsi simulasi sistem kontrol
            score = optimize_func(particle, setpoint=1.0, actual_response=response)

            # Update local and global best
            if score < local_best_scores[i]:
                local_best_scores[i] = score
                local_best_positions[i] = particle

            if score < global_best_score:
                global_best_score = score
                global_best_position = particle

        # Update kecepatan dan posisi partikel
        for i in range(num_particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (local_best_positions[i] - particles[i])
                + c2 * r2 * (global_best_position - particles[i])
            )
            particles[i] = np.clip(particles[i] + velocities[i], [b[0] for b in bounds], [b[1] for b in bounds])

        print(f"Iterasi {iteration + 1}/{max_iter}, Global Best Score: {global_best_score:.4f}")

    return global_best_position, global_best_score

# Fungsi simulasi sistem kontrol sederhana
def simulate_system(params, steps=100):
    Kp, Ki, Kd = params
    setpoint = 1.0
    response = np.zeros(steps)
    integral = 0
    prev_error = 0

    for t in range(steps):
        error = setpoint - response[t - 1] if t > 0 else setpoint
        integral += error
        derivative = error - prev_error

        # PID formula
        response[t] = Kp * error + Ki * integral + Kd * derivative
        prev_error = error

        # Simulasi saturasi output
        response[t] = np.clip(response[t], 0, 1.5)

    return response

# Main
if __name__ == "__main__":
    bounds = [(0, 100), (0, 50), (0, 10)]  # Batas untuk Kp, Ki, Kd
    optimal_params, best_score = pso(fitness_function, bounds)

    print("Parameter PID Optimal:")
    print(f"Kp: {optimal_params[0]:.2f}, Ki: {optimal_params[1]:.2f}, Kd: {optimal_params[2]:.2f}")

    # Plot respon sistem dengan parameter optimal
    optimal_response = simulate_system(optimal_params)
    plt.plot(optimal_response)
    plt.title("Respon Sistem dengan PID Optimal")
    plt.xlabel("Waktu (iterasi)")
    plt.ylabel("Output")
    plt.grid()
    plt.show()
