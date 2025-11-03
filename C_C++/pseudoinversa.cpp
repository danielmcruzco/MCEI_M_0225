
#include <iostream>
#include <Eigen/Dense>

Eigen::MatrixXd pseudoInverse(const Eigen::MatrixXd &A, double tol = 1e-6) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const auto &singularValues = svd.singularValues();
    Eigen::VectorXd singularValuesInv(singularValues.size());

    for (int i = 0; i < singularValues.size(); ++i) {
        singularValuesInv(i) = (singularValues(i) > tol) ? 1.0 / singularValues(i) : 0;
    }

    return svd.matrixV() * singularValuesInv.asDiagonal() * svd.matrixU().transpose();
}

int main() {
    Eigen::MatrixXd A(4, 3);
    A << 1, 0, 2,
         2, -1, 5,
         0, 1, -1,
         1, 3, -1;

    Eigen::MatrixXd A_pinv = pseudoInverse(A);

    std::cout << "Matriz A:\n" << A << "\n";
    std::cout << "Pseudoinversa A+:\n" << A_pinv << "\n\n";

    // Propiedad 1: AA⁺A ≈ A
    std::cout << "AA⁺A:\n" << A * A_pinv * A << "\n\n";

    // Propiedad 2: A⁺AA⁺ ≈ A⁺
    std::cout << "A⁺AA⁺:\n" << A_pinv * A * A_pinv << "\n\n";

    // Propiedad 3: (AA⁺)^T ≈ AA⁺
    std::cout << "(AA⁺)^T:\n" << (A * A_pinv).transpose() << "\n";
    std::cout << "AA⁺:\n" << A * A_pinv << "\n\n";

    // Propiedad 4: (A⁺A)^T ≈ A⁺A
    std::cout << "(A⁺A)^T:\n" << (A_pinv * A).transpose() << "\n";
    std::cout << "A⁺A:\n" << A_pinv * A << "\n";

    return 0;
}

