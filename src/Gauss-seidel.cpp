/*Karen González Ramírez 22110358
Rafael Martinez Cerda 22110385 
Proyecto de Gauss-Seidel 
16 de Junio del 2023*/

#include <iostream>
#include <Eigen/Dense>

class GaussSeidelSolver {
private:
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    Eigen::VectorXd x;
    int maxIterations;
    double tolerance;

public:
    GaussSeidelSolver(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, int maxIterations, double tolerance)
        : A(A), b(b), maxIterations(maxIterations), tolerance(tolerance) {}

    Eigen::VectorXd solve() {
        int n = A.rows();
        x = Eigen::VectorXd::Zero(n);

        for (int iteration = 0; iteration < maxIterations; ++iteration) {
            Eigen::VectorXd xNew = Eigen::VectorXd::Zero(n);

            for (int i = 0; i < n; ++i) {
                double sigma = 0.0;

                for (int j = 0; j < n; ++j) {
                    if (j != i)
                        sigma += A(i, j) * x(j);
                }

                xNew(i) = (b(i) - sigma) / A(i, i);
            }

            double error = (xNew - x).norm();

            if (error < tolerance) {
                std::cout << "Converged after " << iteration + 1 << " iterations." << std::endl;
                return xNew;
            }

            x = xNew;
        }

        std::cout << "Did not converge after " << maxIterations << " iterations." << std::endl;
        return x;
    }
};

int main() {
    Eigen::MatrixXd A(3, 3);
    A << 4, -1, 0,
         -1, 4, -1,
         0, -1, 4;

    Eigen::VectorXd b(3);
    b << 2, 6, 7;

    int maxIterations = 100;
    double tolerance = 1e-6;

    GaussSeidelSolver solver(A, b, maxIterations, tolerance);
    Eigen::VectorXd solution = solver.solve();

    std::cout << "Solution:\n" << solution << std::endl;

    return 0;
}
