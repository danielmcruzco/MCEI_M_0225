
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

// 1. Inversa de Moore penrose
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

// 2. Sistemas de ecuaciones lineales

Eigen::MatrixXd B(4, 4);
    B << 1, 4, 2, 3,
         0, 1, 4, 4,
        -1, 0, 1, 0,
         2, 0, 4, 1;

    // Vector de términos independientes
    Eigen::VectorXd c(4);
    c << 1, 2, 3, 4;

    // Eliminación de Gauss
    Eigen::VectorXd x_gauss = B.partialPivLu().solve(c);
    std::cout << "Solución por Eliminación de Gauss:\n" << x_gauss << "\n\n";

    // Factorización LU
    Eigen::PartialPivLU<Eigen::MatrixXd> lu(B);
    Eigen::VectorXd x_lu = lu.solve(c);
    std::cout << "Solución por LU:\n" << x_lu << "\n";

    // Factorización QR
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(B);
    Eigen::MatrixXd Q = qr.householderQ();
    Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();

    std::cout << "Matriz Q:\n" << Q << "\n\n";
    std::cout << "Matriz R:\n" << R << "\n\n";

    // Resolver el sistema usando QR
    Eigen::VectorXd x_qr = qr.solve(c);
    std::cout << "Solución por QR:\n" << x_qr << "\n";

    // 3. Condicionamiento numérico y estabilidad

    Eigen::MatrixXd D(2, 2);
    D << 2, 1,
         201, 1;

    // Vector original
    Eigen::VectorXd e(2);
    e << 2, 7;

    // Vector perturbado

    Eigen::VectorXd e_perturbado(2);
    e_perturbado << 2.01, 7; // pequeña perturbación en la primera ecuación


    std::cout << "Matriz A:\n" << D << "\n\n";

    // Calcular el número de condición usando valores singulares
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(D);
    double sigma_max = svd.singularValues()(0);
    double sigma_min = svd.singularValues()(svd.singularValues().size() - 1);
    double cond_number = sigma_max / sigma_min;

    std::cout << "Valores singulares:\n" << svd.singularValues() << "\n";
    std::cout << "Número de condición (norma 2): " << cond_number << "\n";


  // Resolver sistema original
    Eigen::VectorXd x_original = D.fullPivLu().solve(e);


 // Resolver sistema perturbado
    Eigen::VectorXd x_perturbado = D.fullPivLu().solve(e_perturbado);


   std::cout << "Solución original:\n" << x_original << "\n\n";
    std::cout << "Solución con perturbación:\n" << x_perturbado << "\n\n";

    // Calcular error relativo
    double error_relativo = (x_perturbado - x_original).norm() / x_original.norm();
    std::cout << "Error relativo en la solución: " << error_relativo << "\n";


    return 0;
}


