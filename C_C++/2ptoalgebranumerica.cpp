
#include <iostream>
#include <Eigen/Dense>
#include <chrono>

int main() {
    using namespace std::chrono;

    // Sistema de ecuaciones: Ax = b
    Eigen::MatrixXd A(4, 4);
    A << 1, 4, 2, 3,
         0, 1, 4, 4,
        -1, 0, 1, 0,
         2, 0, 4, 1;

    Eigen::VectorXd b(4);
    b << 1, 2, 3, 4;

    std::cout << "Matriz A:\n" << A << "\n";
    std::cout << "Vector b:\n" << b << "\n\n";

    // ============================
    // Método 1: Eliminación de Gauss
    // ============================
    auto start_gauss = high_resolution_clock::now();
    Eigen::VectorXd x_gauss = A.fullPivLu().solve(b);
    auto end_gauss = high_resolution_clock::now();
    double time_gauss = duration<double, std::milli>(end_gauss - start_gauss).count();
    double error_gauss = (A * x_gauss - b).norm();


std::cout << "Solución por Gauss:\n" << x_gauss << "\n";
    std::cout << "Tiempo Gauss: " << time_gauss << " ms\n";
    std::cout << "Error Gauss: " << error_gauss << "\n\n";

    // ============================
    // Método 2: Factorización LU
    // ============================
    auto start_lu = high_resolution_clock::now();
    Eigen::PartialPivLU<Eigen::MatrixXd> lu(A);
    Eigen::VectorXd x_lu = lu.solve(b);

  auto end_lu = high_resolution_clock::now();
    double time_lu = duration<double, std::milli>(end_lu - start_lu).count();
    double error_lu = (A * x_lu - b).norm();

    std::cout << "Solución por LU:\n" << x_lu << "\n";
    std::cout << "Tiempo LU: " << time_lu << " ms\n";
    std::cout << "Error LU: " << error_lu << "\n\n";

    // ============================
    // Método 3: Factorización QR

    // ============================
    auto start_qr = high_resolution_clock::now();
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
    Eigen::VectorXd x_qr = qr.solve(b);
    auto end_qr = high_resolution_clock::now();
    double time_qr = duration<double, std::milli>(end_qr - start_qr).count();
    double error_qr = (A * x_qr - b).norm();

    std::cout << "Solución por QR:\n" << x_qr << "\n";
    std::cout << "Tiempo QR: " << time_qr << " ms\n";

    std::cout << "Error QR: " << error_qr << "\n\n";

    return 0;
}

