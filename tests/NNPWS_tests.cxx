#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <functional>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <random>

#include "NNPWS.hxx"

#ifdef NNPWS_USE_OPENMP
#include <omp.h>
#endif

int g_tests_passed = 0;
int g_tests_failed = 0;
double g_abs_tol = 1e-4;

#define COL_RED "\033[1;31m"
#define COL_GREEN "\033[1;32m"
#define COL_YELLOW "\033[1;33m"
#define COL_CYAN "\033[1;36m"
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

struct IAPWS_Point {
    double P; // MPa
    double T; // K
    double v; // m3/kg
    double s; // kJ/kg.K
    double cp;// kJ/kg.K
    double kappa; // 1/MPa
};

NNPWS nnpwsPT(PT,"../resources/models/DNN_TP_v8.pt", std::nullopt);
NNPWS nnpwsPH(PH,"../resources/models/DNN_TP_v8.pt", "../resources/models/DNN_Backward_PH_noRegion.pt");

void check_point(const IAPWS_Point& ref, double d1, double d2) {
    nnpwsPT.setPT(ref.P, ref.T);

    if (!nnpwsPT.isValid()) {
        std::cout << COL_RED "[FAIL] Point invalide (Hors region) : P=" << ref.P << " T=" << ref.T << COL_RESET << std::endl;
        g_tests_failed++;
        return;
    }

    // 1. Densité
    double rho_ref = 1.0 / ref.v;
    double rho_calc = nnpwsPT.getDensity();
    EXPECT_NEAR(rho_calc, rho_ref, d1);

    // 2. Entropie
    double s_calc = nnpwsPT.getEntropy();
    EXPECT_NEAR(s_calc, ref.s, d1);

    // 3. Cp
    double cp_calc = nnpwsPT.getCp();
    EXPECT_NEAR(cp_calc, ref.cp, d2);
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

double acc = 1e-3;

TEST(Region1, Point_300K_3MPa_PH) {
    nnpwsPT.setPT(3.0, 300.0);
    nnpwsPH.setPH(3.0, nnpwsPT.getEnthalpy());

    EXPECT_NEAR(nnpwsPH.getDensity(), nnpwsPT.getDensity(), acc);
    EXPECT_NEAR(nnpwsPH.getEntropy(), nnpwsPT.getEntropy(), acc);
    EXPECT_NEAR(nnpwsPH.getCp(), nnpwsPT.getCp(), acc);
}

TEST(Region1, Point_500K_3MPa_PH) {
    nnpwsPT.setPT(3.0, 500.0);
    nnpwsPH.setPH(3.0, nnpwsPT.getEnthalpy());

    EXPECT_NEAR(nnpwsPH.getDensity(), nnpwsPT.getDensity(), acc);
    EXPECT_NEAR(nnpwsPH.getEntropy(), nnpwsPT.getEntropy(), acc);
    EXPECT_NEAR(nnpwsPH.getCp(), nnpwsPT.getCp(), acc);
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

TEST(Region2, Point_800K_8MPa_PH) {
    nnpwsPT.setPT(8.0, 800.0);
    nnpwsPH.setPH(8.0, nnpwsPT.getEnthalpy());

    EXPECT_NEAR(nnpwsPH.getDensity(), nnpwsPT.getDensity(), acc);
    EXPECT_NEAR(nnpwsPH.getEntropy(), nnpwsPT.getEntropy(), acc);
    EXPECT_NEAR(nnpwsPH.getCp(), nnpwsPT.getCp(), acc);
}

TEST(Region2, Point_650K_0_1MPa_PH) {
    nnpwsPT.setPT(0.1, 650.0);
    nnpwsPH.setPH(0.1, nnpwsPT.getEnthalpy());

    EXPECT_NEAR(nnpwsPH.getDensity(), nnpwsPT.getDensity(), acc);
    EXPECT_NEAR(nnpwsPH.getEntropy(), nnpwsPT.getEntropy(), acc);
    EXPECT_NEAR(nnpwsPH.getCp(), nnpwsPT.getCp(), acc);
}

TEST(Region2, Point_1070K_15MPa_PH) {
    nnpwsPT.setPT(15.0, 1070.0);
    nnpwsPH.setPH(15.0, nnpwsPT.getEnthalpy());

    EXPECT_NEAR(nnpwsPH.getDensity(), nnpwsPT.getDensity(), acc);
    EXPECT_NEAR(nnpwsPH.getEntropy(), nnpwsPT.getEntropy(), acc);
    EXPECT_NEAR(nnpwsPH.getCp(), nnpwsPT.getCp(), acc);
}


// TESTS BATCH CONSISTENCY

TEST(Systeme, BatchConsistencyPT) {
    const std::vector<double> P = {3.0, 8.0, 1.0, 15.0};
    const std::vector<double> T = {300.0, 600.0, 400.0, 700.0};
    std::vector<NNPWS> res;

    NNPWS::compute_batch_PT(P, T, res, "../resources/models/DNN_TP_v8.pt");

    for(size_t i=0; i<P.size(); ++i) {
        nnpwsPT.setPT(P.at(i), T.at(i));
        if(nnpwsPT.isValid() && res[i].isValid()) {
            EXPECT_NEAR(nnpwsPT.getDensity(), res[i].getDensity(), 1e-5);
            EXPECT_NEAR(nnpwsPT.getCp(), res[i].getCp(), 1e-5);
        }
    }
}

TEST(Systeme, BatchConsistencyPH) {
    const std::vector<double> P = {3.0, 8.0, 1.0, 15.0};
    const std::vector<double> H = {115.33127302143888, 2905.5319195853235, 533.4632679456029, 3078.8476570556554};

    std::vector<NNPWS> res;

    NNPWS::compute_batch_PH(P, H, res, "../resources/models/DNN_TP_v8.pt", "../resources/models/DNN_Backward_PH_noRegion.pt");

    for(size_t i=0; i<P.size(); ++i) {
        nnpwsPH.setPH(P.at(i), H.at(i));
        if(nnpwsPH.isValid() && res[i].isValid()) {
            EXPECT_NEAR(nnpwsPH.getDensity(), res[i].getDensity(), 1e-5);
            EXPECT_NEAR(nnpwsPH.getCp(), res[i].getCp(), 1e-5);
        }
    }
}


// PERFORMANCE BENCHMARKS

TEST(Performance, SpeedTest) {
    const int N_SAMPLES = 1000000;
    std::cout << "\n\n" << COL_CYAN << "[BENCHMARK] Generating " << N_SAMPLES << " points..." << COL_RESET << std::endl;

    std::vector<double> P_vec(N_SAMPLES);
    std::vector<double> T_vec(N_SAMPLES);

    std::mt19937 gen(42);
    std::uniform_real_distribution<> disP(1.0, 20.0);
    std::uniform_real_distribution<> disT(300.0, 800.0);

    for(int i=0; i<N_SAMPLES; ++i) {
        P_vec[i] = disP(gen);
        T_vec[i] = disT(gen);
    }

    NNPWS perf_ws(PT, "../resources/models/DNN_TP_v8.pt", std::nullopt);
    std::vector<NNPWS> batch_results;

    typedef std::chrono::high_resolution_clock Clock;
    auto tic = Clock::now();
    auto toc = Clock::now();
    double duration_setPT_noOMP = 0.0;
    double duration_setPT_OMP = 0.0;
    double duration_Batch_noOMP = 0.0;
    double duration_Batch_OMP = 0.0;

    int max_threads = 1;
#ifdef NNPWS_USE_OPENMP
    max_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif

    tic = Clock::now();
    for(int i=0; i<N_SAMPLES; ++i) {
        perf_ws.setPT(P_vec[i], T_vec[i]);
        volatile double d = perf_ws.getDensity();
    }
    toc = Clock::now();
    duration_setPT_noOMP = std::chrono::duration<double>(toc - tic).count();

#ifdef NNPWS_USE_OPENMP
    omp_set_num_threads(max_threads);
#endif

    tic = Clock::now();
    for(int i=0; i<N_SAMPLES; ++i) {
        perf_ws.setPT(P_vec[i], T_vec[i]);
        volatile double d = perf_ws.getDensity();
    }
    toc = Clock::now();
    duration_setPT_OMP = std::chrono::duration<double>(toc - tic).count();

#ifdef NNPWS_USE_OPENMP
    omp_set_num_threads(1);
#endif

    tic = Clock::now();
    NNPWS::compute_batch_PT(P_vec, T_vec, batch_results, "../resources/models/DNN_TP_v8.pt");
    toc = Clock::now();
    duration_Batch_noOMP = std::chrono::duration<double>(toc - tic).count();

#ifdef NNPWS_USE_OPENMP
    omp_set_num_threads(max_threads);
#endif

    tic = Clock::now();
    NNPWS::compute_batch_PT(P_vec, T_vec, batch_results, "../resources/models/DNN_TP_v8.pt");
    toc = Clock::now();
    duration_Batch_OMP = std::chrono::duration<double>(toc - tic).count();

    std::cout << std::fixed << std::setprecision(4);
    std::cout << COL_YELLOW << "========================================================" << std::endl;
    std::cout << " RESULTATS PERFORMANCE (" << N_SAMPLES << " echantillons)" << std::endl;
    std::cout << "========================================================" << COL_RESET << std::endl;

    std::cout << std::left << std::setw(30) << "Methode"
              << std::setw(15) << "Threads"
              << std::setw(15) << "Temps (s)"
              << "Vitesse (pts/s)" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    auto print_row = [&](std::string name, std::string threads, double t) {
        std::cout << std::left << std::setw(30) << name
                  << std::setw(15) << threads
                  << std::setw(15) << t
                  << (int)(N_SAMPLES/t) << std::endl;
    };

    print_row("setPT (Iteratif C++)", "1", duration_setPT_noOMP);
#ifdef NNPWS_USE_OPENMP
    print_row("setPT (Iteratif C++)", std::to_string(max_threads), duration_setPT_OMP);
#else
    print_row("setPT (Iteratif C++)", "N/A (No OMP)", duration_setPT_OMP);
#endif

    std::cout << "--------------------------------------------------------" << std::endl;

    print_row("compute_batch_PT (Torch)", "1", duration_Batch_noOMP);
#ifdef NNPWS_USE_OPENMP
    print_row("compute_batch_PT (Torch)", std::to_string(max_threads), duration_Batch_OMP);
#else
    print_row("compute_batch_PT (Torch)", "N/A (No OMP)", duration_Batch_OMP);
#endif

    std::cout << COL_YELLOW << "========================================================" << COL_RESET << std::endl;

    double gain_batch_vs_iter = duration_setPT_noOMP / duration_Batch_noOMP;
    std::cout << ">>> Gain Batch vs Iteratif: x" << gain_batch_vs_iter << std::endl;
#ifdef NNPWS_USE_OPENMP
    double gain_batch_vs_iter_OMP = duration_setPT_noOMP / duration_Batch_OMP;
    std::cout << ">>> Gain Batch vs Iteratif avec OMP: x" << gain_batch_vs_iter_OMP << std::endl;

    double gain_omp_iter = duration_setPT_noOMP / duration_setPT_OMP;
    std::cout << ">>> Gain OpenMP sur Iteratif: x" << gain_omp_iter
              << (gain_omp_iter < 1.0 ? " (Ralentissement du au surcout threads)" : "") << std::endl;

    double gain_omp_batch = duration_Batch_noOMP / duration_Batch_OMP;
    std::cout << ">>> Gain OpenMP sur Batch: x" << gain_omp_batch
              << (gain_omp_batch < 1.0 ? " (Ralentissement du au surcout threads)" : "") << std::endl;
#endif

    g_tests_passed++;
}

int main(int argc, char** argv) {
    NNPWS::setUseGPU(false);
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