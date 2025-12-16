#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <functional>
#include <algorithm>
#include <iomanip>
#include "NNPWS.hxx"

int g_tests_passed = 0;
int g_tests_failed = 0;
double g_abs_tol = 1e-4;

#define COL_RED "\033[1;31m"
#define COL_GREEN "\033[1;32m"
#define COL_RESET "\033[0m"

#define TEST(Group, Name) \
    void Group##_##Name(); \
    int call_##Group##_##Name = (register_test(#Group "_" #Name, Group##_##Name), 0); \
    void Group##_##Name()

std::vector<std::pair<std::string, std::function<void()>>> g_test_registry;

void register_test(const std::string& name, const std::function<void()>& func) {
    g_test_registry.emplace_back(name, func);
}


#define EXPECT_NEAR(val, ref, rel_tol) \
    do { \
        double v = (val); double r = (ref); double rt = (rel_tol); \
        double diff = std::abs(v - r); \
        double abs_ref = std::abs(r); \
        bool passed = false; \
        double calc_rel_err = 0.0; \
        if (abs_ref > 1e-5) { \
            calc_rel_err = diff / abs_ref; \
            if (calc_rel_err <= rt) passed = true; \
        } \
        if (!passed && diff <= g_abs_tol) passed = true; \
        \
        if (!passed) { \
            std::cerr << COL_RED "[FAIL\t] " << __FILE__ << ":" << __LINE__ \
                      << " | Val: " << v << " Ref: " << r \
                      << "\n\t  > RelErr: " << (calc_rel_err*100.0) << "% (Max " << (rt*100.0) << "%)" \
                      << "\n\t  > AbsErr: " << diff << " (Max " << g_abs_tol << ")" COL_RESET << std::endl; \
            g_tests_failed++; \
        } else { \
            g_tests_passed++; \
        } \
    } while(0)

#define ASSERT_TRUE(condition) \
    do { \
        if (!(condition)) { \
            std::cerr << COL_RED "[FATAL\t] " << __FILE__ << ":" << __LINE__ \
                      << " | Assertion failed: " #condition COL_RESET << std::endl; \
            g_tests_failed++; \
            return; \
        } else { \
            g_tests_passed++; \
        } \
    } while(0)

struct IAPWS_Point {
    double P; // MPa
    double T; // K
    double v; // m3/kg
    double s; // kJ/kg.K
    double cp;// kJ/kg.K
    double kappa; // 1/MPa
};

NNPWS nnpws("../resources/models/DNN_TP_v8.pt", "");

void check_point(const IAPWS_Point& ref, double d1, double d2) {
    //NNPWS w(inputPair::PT, ref.P, ref.T, "../resources/models/DNN_TP_v6.pt", "");
    nnpws.setPT(ref.P, ref.T);

    if (!nnpws.isValid()) {
        std::cout << COL_RED "[FAIL] Point invalide (Hors region) : P=" << ref.P << " T=" << ref.T << COL_RESET << std::endl;
        g_tests_failed++;
        return;
    }

    // 1. Densité
    double rho_ref = 1.0 / ref.v;
    double rho_calc = nnpws.getDensity();
    EXPECT_NEAR(rho_calc, rho_ref, d1);

    // 2. Entropie
    double s_calc = nnpws.getEntropy();
    EXPECT_NEAR(s_calc, ref.s, d1);

    // 3. Cp
    double cp_calc = nnpws.getCp();
    EXPECT_NEAR(cp_calc, ref.cp, d2);

    // 4. kappa
    double kappa_calc = nnpws.getKappa();
    EXPECT_NEAR(kappa_calc, ref.kappa, d2);
}


// SETUP
bool global_setup(const std::string& model_path) {
    static bool loaded = false;
    if (loaded) return true;
    std::cout << "[SETUP] Chargement du modele : " << model_path << std::endl;

    g_abs_tol = 1e-2;
    //if (NNPWS::init(model_path) != 0) {
    //    std::cerr << COL_RED "FATAL: Impossible de charger le fichier .pt" COL_RESET << std::endl;
    //    return false;
    //}
    loaded = true;
    return true;
}

// TESTS REGION 1

TEST(Region1, Point_300K_3MPa) {
    check_point({3.0, 300.0, 0.0010021516796866943,  0.39229479240262577, 4.173012184067787, 0.00044638212280219354}, 1, 1);
}

TEST(Region1, Point_300K_80MPa) {
    check_point({80.0, 300.0, 0.0009711808940216298, 0.3685638523984814, 4.010089869646329, 0.0003720394372317089}, 1, 1);
}

TEST(Region1, Point_500K_3MPa) {
    check_point({3.0, 500.0, 0.001202418003378339, 2.5804191200518094, 4.6558068221112086, 0.0011289218770058733}, 2, 1);
}

// TESTS REGION 2

TEST(Region2, Point_800K_8MPa) {
    check_point({8, 800.0, 0.043613196528163034, 6.810345602358012, 2.4445995377277243, 0.13256964210683342}, 1, 1);
}

TEST(Region2, Point_650K_0_1MPa) {
    check_point({0.1, 650.0, 2.9954568908865418, 8.472894377244847, 2.0554762342977186, 10.014916019781394}, 1, 1);
}

TEST(Region2, Point_1070K_15MPa) {
    check_point({15, 1070.0, 0.03201002003375847, 7.196542902382565, 2.515290135706759, 0.06854017478483342}, 1, 1);
}

// TESTS BATCH

TEST(Systeme, BatchConsistency) {
    const std::vector<double> P = {3.0, 80.0, 0.0035, 30.0};
    const std::vector<double> T = {300.0, 300.0, 300.0, 700.0};
    std::vector<NNPWS> res;

    NNPWS::compute_batch_PT(P, T, res, "../resources/models/DNN_TP_v8.pt");

    for(size_t i=0; i<P.size(); ++i) {
        nnpws.setPT(P.at(i), T.at(i));
        if(nnpws.isValid() && res[i].isValid()) {
            EXPECT_NEAR(nnpws.getDensity(), res[i].getDensity(), 1e-5);
            EXPECT_NEAR(nnpws.getCp(), res[i].getCp(), 1e-5);
        }
    }
}

int main(int argc, char** argv) {
    //if (!global_setup("../models/DNN_TP_v6.pt")) return -1;

    std::string filter = (argc > 1) ? argv[1] : "";
    if (!filter.empty()) std::cout << ">>> FILTRE: " << filter << std::endl;

    std::cout << "========================================================" << std::endl;
    int run_count = 0;

    for (const auto& test : g_test_registry) {
        if (!filter.empty() && test.first.find(filter) == std::string::npos) continue;

        run_count++;
        int fail_pre = g_tests_failed;
        std::cout << "[RUN\t\t] " << std::left << std::setw(35) << test.first << std::flush;

        test.second();

        if (g_tests_failed > fail_pre) {
            std::cout << "\r" << COL_RED << "[FAILED\t] " << test.first << std::string(15, ' ') << COL_RESET << std::endl;
        } else {
            std::cout << "\r" << COL_GREEN << "[OK\t\t] " << test.first << std::string(15, ' ') << COL_RESET << std::endl;
        }
    }

    std::cout << "========================================================" << std::endl;
    std::cout << "Tests: " << run_count << " | Passed: " << g_tests_passed << std::endl;

    if (g_tests_failed > 0) {
        std::cout << COL_RED "FAILURES: " << g_tests_failed << COL_RESET << std::endl;
        return 1;
    }
    std::cout << COL_GREEN "SUCCESS" COL_RESET << std::endl;
    return 0;
}