#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <mpi.h>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <climits>
#include <ctime>
#include <string>
#include <random>

// Nombre que aparece en consola y logs
#define NOMBRE_EQUIPO "BRANDON ERNIE MEDINA ROBLEDO"

// ----------------------------------------------------------------------------
// Contexto global por proceso
// ----------------------------------------------------------------------------
struct Contexto {
    int rank;
    int size;
    int hilos;
    int N;          // tamano global del arreglo (conocido por todos los nodos)
    bool verbose;   // impresiones detalladas
    std::string pc;
    FILE* log;
    Contexto() : rank(0), size(1), hilos(1), N(0), verbose(false), log(NULL) {}
};
static Contexto G;

static int* datos = NULL;       // arreglo completo (solo nodo 0)
static bool ordenado = false;   // solo nodo 0
static int ultimoMax = 0;       // maximo del rango de la ultima generacion (nodo 0)

struct Params {
    int opcion, N, minV, maxV, seed, verbose, modo, target, reps;
};

struct Metricas {
    double sortSeq, sortPar;
    double secLoc1, secLocT, secDist, secDistComp, secDistComm;
    double binLoc1, binLocT, binDist, binDistComp, binDistComm;
    int procsDist;
    Metricas() {
        sortSeq = sortPar = -1;
        secLoc1 = secLocT = secDist = secDistComp = secDistComm = -1;
        binLoc1 = binLocT = binDist = binDistComp = binDistComm = -1;
        procsDist = 0;
    }
};
static Metricas M;

// ----------------------------------------------------------------------------
// Utilidades de formato y registro (log)
// ----------------------------------------------------------------------------
static std::string fmt(const char* f, ...) {
    va_list ap, ap2;
    va_start(ap, f);
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, f, ap);
    va_end(ap);
    if (n < 0) n = 0;
    std::string s;
    s.resize((size_t)n + 1);
    vsnprintf(&s[0], (size_t)n + 1, f, ap2);
    va_end(ap2);
    s.resize((size_t)n);
    return s;
}

static std::string ahora() {
    time_t t = time(NULL);
    struct tm tmv;
#ifdef _WIN32
    localtime_s(&tmv, &t);
#else
    localtime_r(&t, &tmv);
#endif
    char* buf = new char[32];
    strftime(buf, 32, "%Y-%m-%d %H:%M:%S", &tmv);
    std::string s(buf);
    delete[] buf;
    return s;
}

// Escribe siempre en el log del nodo; en consola solo si 'consola' es true.
static void emitir(bool consola, const std::string& msg) {
    std::string ts = ahora();
#pragma omp critical(seccion_log)
    {
        if (G.log) {
            fprintf(G.log, "[%s] %s\n", ts.c_str(), msg.c_str());
            fflush(G.log);
        }
        if (consola) {
            printf("%s\n", msg.c_str());
            fflush(stdout);
        }
    }
}

// Mensaje generico del nodo
static void info(bool consola, const std::string& msg) {
    emitir(consola, fmt("[Equipo: %s] [PC: %s] [Nodo MPI: %d] %s",
                        NOMBRE_EQUIPO, G.pc.c_str(), G.rank, msg.c_str()));
}

// Mensaje con el formato completo requerido (equipo, nodo, hilo, bloque, resultado)
static void reg(bool consola, int hilo, long long a, long long b,
                const char* algo, const std::string& estado) {
    std::string h = (hilo < 0) ? std::string("-") : std::to_string(hilo);
    emitir(consola, fmt("[Equipo: %s] [PC: %s] [Nodo MPI: %d] [Hilo OpenMP: %s] "
                        "[Bloque: %lld - %lld] [Algoritmo: %s] [Estado: %s]",
                        NOMBRE_EQUIPO, G.pc.c_str(), G.rank, h.c_str(), a, b,
                        algo, estado.c_str()));
}

static void imprimirBloque(const int* a, int n, long long offset,
                           const char* titulo, bool consola) {
    std::string s = fmt("%s (posiciones %lld - %lld): ", titulo, offset, offset + n - 1);
    for (int i = 0; i < n; i++) s += std::to_string(a[i]) + " ";
    info(consola, s);
}

static double ratio(double a, double b) { return (b > 0.0) ? a / b : 0.0; }

// ----------------------------------------------------------------------------
// Ordenamiento: merge sort secuencial y paralelo (OpenMP)
// ----------------------------------------------------------------------------
static void mezclar(int* a, int* tmp, int lo, int mid, int hi) {
    int i = lo, j = mid, k = lo;
    while (i < mid && j < hi) tmp[k++] = (a[i] <= a[j]) ? a[i++] : a[j++];
    while (i < mid) tmp[k++] = a[i++];
    while (j < hi) tmp[k++] = a[j++];
    for (k = lo; k < hi; k++) a[k] = tmp[k];
}

static void mergeSortSeq(int* a, int* tmp, int lo, int hi) {
    if (hi - lo < 2) return;
    int mid = lo + (hi - lo) / 2;
    mergeSortSeq(a, tmp, lo, mid);
    mergeSortSeq(a, tmp, mid, hi);
    mezclar(a, tmp, lo, mid, hi);
}

// Parallel Merge Sort: cada hilo ordena un tramo y luego se mezclan por pares
// (en paralelo en cada nivel). Solo usa construcciones de OpenMP 2.0.
static void mergeSortParalelo(int* a, int n, int hilos, bool det) {
    if (n < 2) return;
    int* tmp = new int[n];
    int chunks = (hilos < n) ? hilos : n;
    if (chunks < 1) chunks = 1;
    int* lim = new int[chunks + 1];
    for (int i = 0; i <= chunks; i++) lim[i] = (int)((long long)n * i / chunks);

#pragma omp parallel for num_threads(hilos) schedule(static)
    for (int i = 0; i < chunks; i++) {
        mergeSortSeq(a, tmp, lim[i], lim[i + 1]);
        if (det)
            reg(G.verbose, omp_get_thread_num(), lim[i], lim[i + 1] - 1,
                "Parallel Merge Sort", "tramo ordenado localmente");
    }

    int runs = chunks;
    while (runs > 1) {
        int pares = runs / 2;
#pragma omp parallel for num_threads(hilos) schedule(static)
        for (int p = 0; p < pares; p++) {
            mezclar(a, tmp, lim[2 * p], lim[2 * p + 1], lim[2 * p + 2]);
        }
        int nuevos = (runs + 1) / 2;
        for (int p = 0; p < nuevos; p++) lim[p] = lim[2 * p];
        lim[nuevos] = n;
        runs = nuevos;
    }
    delete[] lim;
    delete[] tmp;
}

static bool estaOrdenado(const int* a, int n) {
    for (int i = 1; i < n; i++)
        if (a[i - 1] > a[i]) return false;
    return true;
}

// ----------------------------------------------------------------------------
// Busquedas locales a un bloque (usadas por el esquema local y por cada nodo)
// Devuelven el indice GLOBAL (indice local + offset) o -1.
// ----------------------------------------------------------------------------
static int buscarSecuencial(const int* a, int n, int target, long long off,
                            int hilos, bool det) {
    const char* algo = "Busqueda Secuencial";
    if (n <= 0) return -1;
    int* res = new int[hilos];
    for (int i = 0; i < hilos; i++) res[i] = -1;
    bool porElem = det && n <= 100;

#pragma omp parallel num_threads(hilos)
    {
        int t = omp_get_thread_num();
        int nt = omp_get_num_threads();
        int ini = (int)((long long)n * t / nt);
        int fin = (int)((long long)n * (t + 1) / nt);
        for (int i = ini; i < fin; i++) {
            if (porElem)
                reg(G.verbose, t, off + ini, off + fin - 1, algo,
                    fmt("revisa pos %lld valor %d (%s)", off + i, a[i],
                        a[i] == target ? "coincide" : "no coincide"));
            if (a[i] == target) { res[t] = i; break; }
        }
        if (det) {
            if (ini >= fin)
                reg(G.verbose, t, off + ini, off + fin - 1, algo, "bloque vacio");
            else if (res[t] >= 0)
                reg(G.verbose, t, off + ini, off + fin - 1, algo,
                    fmt("Elemento Encontrado en Indice Global: %lld", off + res[t]));
            else
                reg(G.verbose, t, off + ini, off + fin - 1, algo,
                    "Elemento no encontrado en este hilo");
        }
    }
    int mejor = -1;
    for (int i = 0; i < hilos; i++)
        if (res[i] >= 0 && (mejor < 0 || res[i] < mejor)) mejor = res[i];
    delete[] res;
    return (mejor < 0) ? -1 : (int)(mejor + off);
}

static int buscarBinaria(const int* a, int n, int target, long long off,
                         int hilos, bool det) {
    const char* algo = "Busqueda Binaria";
    if (n <= 0) return -1;
    // Evaluacion de rango del bloque (primer y ultimo elemento)
    if (target < a[0] || target > a[n - 1]) {
        if (det)
            reg(G.verbose, -1, off, off + n - 1, algo,
                fmt("valor %d fuera del rango del bloque [%d, %d]: no se busca",
                    target, a[0], a[n - 1]));
        return -1;
    }
    if (det)
        reg(G.verbose, -1, off, off + n - 1, algo,
            fmt("valor %d dentro del rango del bloque [%d, %d]: se busca",
                target, a[0], a[n - 1]));

    int* res = new int[hilos];
    for (int i = 0; i < hilos; i++) res[i] = -1;

#pragma omp parallel num_threads(hilos)
    {
        int t = omp_get_thread_num();
        int nt = omp_get_num_threads();
        int ini = (int)((long long)n * t / nt);
        int fin = (int)((long long)n * (t + 1) / nt);
        if (ini >= fin) {
            if (det) reg(G.verbose, t, off + ini, off + fin - 1, algo, "subintervalo vacio");
        } else if (target < a[ini] || target > a[fin - 1]) {
            if (det)
                reg(G.verbose, t, off + ini, off + fin - 1, algo,
                    "valor fuera de este subintervalo");
        } else {
            int lo = ini, hi = fin - 1;           // primera ocurrencia
            while (lo < hi) {
                int mid = lo + (hi - lo) / 2;
                if (a[mid] < target) lo = mid + 1; else hi = mid;
            }
            if (a[lo] == target) res[t] = lo;
            if (det) {
                if (res[t] >= 0)
                    reg(G.verbose, t, off + ini, off + fin - 1, algo,
                        fmt("Elemento Encontrado en Indice Global: %lld", off + res[t]));
                else
                    reg(G.verbose, t, off + ini, off + fin - 1, algo,
                        "Elemento no encontrado en este subintervalo");
            }
        }
    }
    int mejor = -1;
    for (int i = 0; i < hilos; i++)
        if (res[i] >= 0 && (mejor < 0 || res[i] < mejor)) mejor = res[i];
    delete[] res;
    return (mejor < 0) ? -1 : (int)(mejor + off);
}

static int ejecutarBusqueda(bool bin, const int* a, int n, int target,
                            long long off, int hilos, bool det) {
    return bin ? buscarBinaria(a, n, target, off, hilos, det)
               : buscarSecuencial(a, n, target, off, hilos, det);
}

// ----------------------------------------------------------------------------
// Esquema LOCAL multihilo (solo nodo 0, solo OpenMP)
// ----------------------------------------------------------------------------
static void busquedaLocal(bool bin, int target, int reps) {
    const char* algo = bin ? "Binaria Local Multihilo" : "Secuencial Local Multihilo";
    int N = G.N;
    info(true, fmt("--- Esquema LOCAL MULTIHILO | %s | target=%d | reps=%d ---",
                   algo, target, reps));

    double t = MPI_Wtime();
    int idx1 = -1;
    for (int r = 0; r < reps; r++) idx1 = ejecutarBusqueda(bin, datos, N, target, 0, 1, false);
    double t1 = (MPI_Wtime() - t) / reps;

    t = MPI_Wtime();
    int idxT = -1;
    for (int r = 0; r < reps; r++)
        idxT = ejecutarBusqueda(bin, datos, N, target, 0, G.hilos, r == 0 && G.verbose);
    double tT = (MPI_Wtime() - t) / reps;

    if (idxT >= 0)
        reg(true, -1, 0, N - 1, algo, fmt("Elemento Encontrado en Indice Global: %d", idxT));
    else
        reg(true, -1, 0, N - 1, algo, "Elemento NO encontrado");
    info(true, fmt("Local 1 hilo: %.9f s | Local %d hilos: %.9f s | speedup %.2f | eficiencia %.2f (indice 1 hilo: %d)",
                   t1, G.hilos, tT, ratio(t1, tT), ratio(ratio(t1, tT), G.hilos), idx1));
    if (bin) { M.binLoc1 = t1; M.binLocT = tT; }
    else     { M.secLoc1 = t1; M.secLocT = tT; }
}

// ----------------------------------------------------------------------------
// Esquema DISTRIBUIDO multinivel (MPI + OpenMP). Lo ejecutan TODOS los nodos.
// ----------------------------------------------------------------------------
static void busquedaDistribuida(bool bin, int target, int reps) {
    const char* algo = bin ? "Binaria Distribuida" : "Secuencial Distribuida";
    int N = G.N, P = G.size, rank = G.rank;

    int* counts = new int[P];
    int* displs = new int[P];
    int base = N / P, rem = N % P, acum = 0;
    for (int i = 0; i < P; i++) {
        counts[i] = base + (i < rem ? 1 : 0);
        displs[i] = acum;
        acum += counts[i];
    }
    int count = counts[rank];
    long long off = displs[rank];

    if (rank == 0)
        info(true, fmt("--- Esquema DISTRIBUIDO MULTINIVEL | %s | %d procesos x %d hilos | target=%d | reps=%d ---",
                       algo, P, G.hilos, target, reps));

    // 1) Bcast del target
    MPI_Barrier(MPI_COMM_WORLD);
    double t = MPI_Wtime();
    MPI_Bcast(&target, 1, MPI_INT, 0, MPI_COMM_WORLD);
    double tBcast = MPI_Wtime() - t;

    // 2) Distribucion de bloques
    int* bloque = new int[count > 0 ? count : 1];
    MPI_Barrier(MPI_COMM_WORLD);
    t = MPI_Wtime();
    MPI_Scatterv(datos, counts, displs, MPI_INT, bloque, count, MPI_INT, 0, MPI_COMM_WORLD);
    double tScatter = MPI_Wtime() - t;

    reg(G.verbose, -1, off, off + count - 1, algo,
        fmt("bloque recibido: %d elementos | Bcast %.6f s | Scatterv %.6f s",
            count, tBcast, tScatter));
    if (G.verbose && count <= 100 && count > 0)
        imprimirBloque(bloque, count, off, "Contenido del bloque", true);

    // 3) Busqueda local del nodo (OpenMP dentro del bloque)
    MPI_Barrier(MPI_COMM_WORLD);
    t = MPI_Wtime();
    int resLocal = -1;
    for (int r = 0; r < reps; r++) {
        int x = ejecutarBusqueda(bin, bloque, count, target, off, G.hilos, r == 0 && G.verbose);
        if (r == 0) resLocal = x;
    }
    double tComp = (MPI_Wtime() - t) / reps;

    if (resLocal >= 0)
        reg(G.verbose, -1, off, off + count - 1, algo,
            fmt("Elemento Encontrado en Indice Global: %d | computo %.9f s", resLocal, tComp));
    else
        reg(G.verbose, -1, off, off + count - 1, algo,
            fmt("Elemento no encontrado en este nodo | computo %.9f s", tComp));

    // 4) Consolidacion con Reduce (MIN; INT_MAX = no encontrado)
    int loc = (resLocal < 0) ? INT_MAX : resLocal;
    int glob = INT_MAX;
    MPI_Barrier(MPI_COMM_WORLD);
    t = MPI_Wtime();
    MPI_Reduce(&loc, &glob, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    double tReduce = MPI_Wtime() - t;

    double* mine = new double[3];
    double* mx = new double[3];
    mine[0] = tComp; mine[1] = tScatter; mine[2] = tBcast;
    MPI_Reduce(mine, mx, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double comm = mx[1] + mx[2] + tReduce;
        double total = comm + mx[0];
        if (glob != INT_MAX)
            reg(true, -1, 0, N - 1, algo, fmt("RESULTADO GLOBAL: Elemento Encontrado en Indice Global: %d", glob));
        else
            reg(true, -1, 0, N - 1, algo, "RESULTADO GLOBAL: Elemento NO encontrado");
        info(true, fmt("Tiempos (MPI_Wtime): total %.9f s | computo (max nodos) %.9f s | comunicacion %.9f s [Bcast %.6f, Scatterv %.6f, Reduce %.6f]",
                       total, mx[0], comm, mx[2], mx[1], tReduce));
        if (bin) { M.binDist = total; M.binDistComp = mx[0]; M.binDistComm = comm; }
        else     { M.secDist = total; M.secDistComp = mx[0]; M.secDistComm = comm; }
        M.procsDist = P;
    }

    delete[] mine; delete[] mx;
    delete[] bloque; delete[] counts; delete[] displs;
}

// ----------------------------------------------------------------------------
// Generacion y ordenamiento (nodo 0)
// ----------------------------------------------------------------------------
static void generarArreglo(const Params& p) {
    if (datos) { delete[] datos; datos = NULL; }
    ordenado = false;
    double t0 = MPI_Wtime();
    datos = new int[p.N];
    unsigned base = p.seed ? (unsigned)p.seed : (unsigned)time(NULL);
    int mn = p.minV, mx = p.maxV;
#pragma omp parallel
    {
        std::mt19937 rng(base + 7919u * (unsigned)omp_get_thread_num());
        std::uniform_int_distribution<int> dist(mn, mx);
#pragma omp for schedule(static)
        for (int i = 0; i < p.N; i++) datos[i] = dist(rng);
    }
    double t = MPI_Wtime() - t0;
    ultimoMax = mx;
    M = Metricas();   // nuevo arreglo: se reinician las mediciones
    info(true, fmt("Arreglo generado: N=%d, rango [%d, %d], semilla=%u, tiempo %.6f s",
                   p.N, mn, mx, base, t));
    if (G.verbose && p.N <= 100) imprimirBloque(datos, p.N, 0, "Arreglo generado", true);
}

static void ordenarArreglo() {
    int N = G.N;
    info(true, fmt("--- Ordenamiento | N=%d | %d hilos OpenMP ---", N, G.hilos));

    int* copia = new int[N];
    memcpy(copia, datos, sizeof(int) * (size_t)N);
    int* tmp = new int[N];
    double t0 = MPI_Wtime();
    mergeSortSeq(copia, tmp, 0, N);
    double tSeq = MPI_Wtime() - t0;
    delete[] tmp;
    delete[] copia;

    t0 = MPI_Wtime();
    mergeSortParalelo(datos, N, G.hilos, G.verbose);
    double tPar = MPI_Wtime() - t0;

    ordenado = estaOrdenado(datos, N);
    M.sortSeq = tSeq;
    M.sortPar = tPar;
    info(true, fmt("Merge sort secuencial: %.6f s | Parallel Merge Sort (%d hilos): %.6f s | speedup %.2f | eficiencia %.2f | verificacion: %s",
                   tSeq, G.hilos, tPar, ratio(tSeq, tPar), ratio(ratio(tSeq, tPar), G.hilos),
                   ordenado ? "ORDENADO" : "ERROR: NO ORDENADO"));
    if (G.verbose && N <= 100) imprimirBloque(datos, N, 0, "Arreglo ordenado", true);
}

// ----------------------------------------------------------------------------
// Comparativa
// ----------------------------------------------------------------------------
static void mostrarBusqueda(const char* nombre, double l1, double lT,
                            double dTot, double dComp, double dComm) {
    if (l1 >= 0 && lT >= 0)
        info(true, fmt("%s LOCAL: 1 hilo %.9f s | %d hilos %.9f s | speedup %.2f | eficiencia %.2f",
                       nombre, l1, G.hilos, lT, ratio(l1, lT), ratio(ratio(l1, lT), G.hilos)));
    else
        info(true, fmt("%s LOCAL: sin medicion", nombre));
    if (dTot >= 0) {
        info(true, fmt("%s DISTRIBUIDA (%d procesos x %d hilos): total %.9f s | computo %.9f s | comunicacion %.9f s",
                       nombre, M.procsDist, G.hilos, dTot, dComp, dComm));
        if (lT >= 0) {
            double spT = ratio(lT, dTot), spC = ratio(lT, dComp);
            info(true, fmt("   Local(%d hilos) vs Distribuida -> speedup total %.2f (ef. %.2f) | speedup solo computo %.2f (ef. %.2f)",
                           G.hilos, spT, ratio(spT, M.procsDist), spC, ratio(spC, M.procsDist)));
        }
        if (l1 >= 0) {
            double spT = ratio(l1, dTot);
            info(true, fmt("   Local(1 hilo) vs Distribuida -> speedup total %.2f (ef. %.2f con %d hilos totales)",
                           spT, ratio(spT, (double)M.procsDist * G.hilos), M.procsDist * G.hilos));
        }
    } else {
        info(true, fmt("%s DISTRIBUIDA: sin medicion", nombre));
    }
}

static void mostrarComparativa() {
    info(true, "================ TIEMPOS Y COMPARATIVA DE RENDIMIENTO ================");
    info(true, fmt("N=%d | hilos OpenMP por proceso=%d", G.N, G.hilos));
    if (M.sortSeq >= 0 && M.sortPar >= 0)
        info(true, fmt("ORDENAMIENTO: secuencial %.6f s | OpenMP %.6f s | speedup %.2f | eficiencia %.2f",
                       M.sortSeq, M.sortPar, ratio(M.sortSeq, M.sortPar),
                       ratio(ratio(M.sortSeq, M.sortPar), G.hilos)));
    else
        info(true, "ORDENAMIENTO: sin medicion");
    mostrarBusqueda("BUSQUEDA SECUENCIAL", M.secLoc1, M.secLocT, M.secDist, M.secDistComp, M.secDistComm);
    mostrarBusqueda("BUSQUEDA BINARIA", M.binLoc1, M.binLocT, M.binDist, M.binDistComp, M.binDistComm);
    info(true, "======================================================================");
}

// ----------------------------------------------------------------------------
// Menu (nodo 0)
// ----------------------------------------------------------------------------
static int leerInt(const char* msg, int def) {
    printf("%s [%d]: ", msg, def);
    fflush(stdout);
    char* buf = new char[64];
    int v = def;
    if (fgets(buf, 64, stdin)) {
        if (buf[0] != '\n' && buf[0] != '\0') v = atoi(buf);
    }
    delete[] buf;
    return v;
}

static Params pedirParametros() {
    Params p;
    memset(&p, 0, sizeof(Params));
    printf("\n=============== MENU - %s ===============\n", NOMBRE_EQUIPO);
    printf(" 1. Generar Arreglo Dinamico\n");
    printf(" 2. Ordenar Arreglo con OpenMP\n");
    printf(" 3. Ejecutar Busqueda Secuencial (Local vs Distribuida)\n");
    printf(" 4. Ejecutar Busqueda Binaria (Local vs Distribuida)\n");
    printf(" 5. Mostrar Tiempos y Comparativa de Rendimiento\n");
    printf(" 6. Salir\n");
    p.opcion = leerInt("Opcion", 6);

    if (p.opcion == 1) {
        p.N = leerInt("Tamano del arreglo", 100);
        if (p.N < 1 || p.N > 2000000000) { printf("Tamano invalido.\n"); p.opcion = 0; return p; }
        p.minV = leerInt("Valor minimo", 0);
        p.maxV = leerInt("Valor maximo", p.N <= 1000 ? 1000 : 100000000);
        if (p.minV > p.maxV) { int x = p.minV; p.minV = p.maxV; p.maxV = x; }
        p.seed = leerInt("Semilla (0 = aleatoria)", 0);
        p.verbose = leerInt("Impresiones detalladas (1=si, 0=no)", p.N <= 1000 ? 1 : 0);
    } else if (p.opcion == 2) {
        if (!datos) { printf("Primero genere el arreglo (opcion 1).\n"); p.opcion = 0; }
    } else if (p.opcion == 3 || p.opcion == 4) {
        bool bin = (p.opcion == 4);
        if (!datos) { printf("Primero genere el arreglo (opcion 1).\n"); p.opcion = 0; return p; }
        if (bin && !ordenado) { printf("La busqueda binaria requiere el arreglo ordenado (opcion 2).\n"); p.opcion = 0; return p; }
        p.modo = leerInt("Esquema: 1=Local multihilo, 2=Distribuido multinivel, 3=Ambos", 3);
        int tipo = leerInt("Target: 1=Manual, 2=Existente al azar, 3=Inexistente", 2);
        if (tipo == 1) p.target = leerInt("Valor a buscar", 0);
        else if (tipo == 3) p.target = ultimoMax + 1;
        else {
            long long r = ((long long)rand() * 32768LL + rand()) % G.N;
            p.target = datos[r];
        }
        int defReps = G.verbose ? 1 : (bin ? 1000 : 5);
        p.reps = leerInt("Repeticiones para promediar tiempos", defReps);
        if (p.reps < 1) p.reps = 1;
    }
    return p;
}

// ----------------------------------------------------------------------------
// main
// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
    int prov = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &prov);
    MPI_Comm_rank(MPI_COMM_WORLD, &G.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &G.size);

    char* nombre = new char[MPI_MAX_PROCESSOR_NAME];
    int len = 0;
    MPI_Get_processor_name(nombre, &len);
    G.pc = nombre;
    delete[] nombre;

    std::string archivo = fmt("log_equipo_%s_nodo_%d.txt", G.pc.c_str(), G.rank);
    G.log = fopen(archivo.c_str(), "a");
    srand((unsigned)time(NULL) + 31u * (unsigned)G.rank);

    int hcfg = 0;
    if (G.rank == 0) hcfg = leerInt("Hilos OpenMP por proceso (0 = automatico)", 0);
    MPI_Bcast(&hcfg, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (hcfg > 0) omp_set_num_threads(hcfg);
    G.hilos = omp_get_max_threads();

    info(true, fmt("===== INICIO DE SESION ===== PC: %s | Rank MPI: %d de %d | Hilos OpenMP: %d | Nucleos logicos detectados: %d | Log: %s",
                   G.pc.c_str(), G.rank, G.size, G.hilos, omp_get_num_procs(), archivo.c_str()));
    MPI_Barrier(MPI_COMM_WORLD);

    bool salir = false;
    while (!salir) {
        Params p;
        memset(&p, 0, sizeof(Params));
        if (G.rank == 0) p = pedirParametros();
        MPI_Bcast(&p, (int)sizeof(Params), MPI_BYTE, 0, MPI_COMM_WORLD);

        switch (p.opcion) {
        case 1:
            G.N = p.N;
            G.verbose = (p.verbose != 0);
            if (G.rank == 0) generarArreglo(p);
            else info(false, fmt("Esperando: el nodo 0 genero el arreglo de %d elementos", p.N));
            break;
        case 2:
            if (G.rank == 0) ordenarArreglo();
            else info(false, "Esperando: el nodo 0 ordena el arreglo con OpenMP");
            break;
        case 3:
        case 4: {
            bool bin = (p.opcion == 4);
            if (p.modo == 1 || p.modo == 3) {
                if (G.rank == 0) busquedaLocal(bin, p.target, p.reps);
                else info(false, "Esquema local: este nodo no participa");
            }
            if (p.modo == 2 || p.modo == 3) busquedaDistribuida(bin, p.target, p.reps);
            break;
        }
        case 5:
            if (G.rank == 0) mostrarComparativa();
            break;
        case 6:
            salir = true;
            info(G.rank == 0, "Fin de sesion");
            break;
        default:
            break;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        fflush(stdout);
    }

    if (datos) delete[] datos;
    if (G.log) fclose(G.log);
    MPI_Finalize();
    return 0;
}
