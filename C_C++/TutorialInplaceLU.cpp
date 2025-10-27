
#include <iostream>
#include <Eigen/Dense>

int main() {
    std::cout << "[init]\n";
    Eigen::MatrixXd A(3, 3);
    A << 3, -0.1, -0.2,
         0.1, 7, -0.3,
         0.3, -0.2, 10;
    std::cout << "Matriz A:\n" << A << "\n";

    std::cout << "[decomposition]\n";
    Eigen::PartialPivLU<Eigen::MatrixXd> lu(A);

    std::cout << "LU (L y U combinadas):\n" << lu.matrixLU() << "\n";
    std::cout << "Permutación P:\n" << lu.permutationP().indices() << "\n";

    std::cout << "[solve]\n";
    Eigen::VectorXd b(3);
    b << 7.85, -19.3, 71.4;
    Eigen::VectorXd x = lu.solve(b);
    
    std::cout << "Solución x:\n" << x << "\n";
    std::cout << "Residual: " << (A * x - b).norm() << "\n";

    return 0;
}
