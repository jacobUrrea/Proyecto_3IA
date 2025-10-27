import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
from collections import defaultdict

class RedBayesiana:
    def __init__(self):
        self.grafo = nx.DiGraph()
        self.cpts = {}
        self.variables = set()

    def cargar_estructura(self, archivo_estructura):
        #Carga la estructura de la red desde CSV
        try:
            df = pd.read_csv(archivo_estructura)

            # Verificar columnas requeridas
            if 'padre' not in df.columns or 'hijo' not in df.columns:
                raise ValueError("El archivo debe contener columnas 'padre' y 'hijo'")

            # Agregar nodos y aristas
            for _, row in df.iterrows():
                padre = row['padre'].strip()
                hijo = row['hijo'].strip()

                self.variables.add(padre)
                self.variables.add(hijo)
                self.grafo.add_edge(padre, hijo)

            print(f"Estructura cargada: {len(self.variables)} variables")
            return True

        except Exception as e:
            print(f"Error cargando estructura: {e}")
            return False
    def cargar_cpt(self, archivo_cpt, nombre_nodo=None):
        #Carga CPT desde archivo específico

        try:
            df = pd.read_csv(archivo_cpt)

            # Detectar padres a partir de las columnas
            columnas = list(df.columns)

            # La ultima columna es siempre 'probabilidad'
            if columnas[-1] != 'probabilidad':
                raise ValueError("La ultima columna debe llamarse 'probabilidad'")

            # Las columnas anteriores son los padres + valor del nodo
            columnas_variables = columnas[:-1]

            # El valor del nodo actual es la penúltima columna
            columna_valor = columnas_variables[-1]

            # Los padres son todas las columnas excepto la ultima y probabilidad
            padres = columnas_variables[:-1] if len(columnas_variables) > 1 else []

            # Verificar que los padres existan en el grafo
            for padre in padres:
                if padre not in self.grafo:
                    print(f"Advertencia: {padre} no esta en la estructura de la red")

            # Almacenar CPT con metadatos
            self.cpts[nombre_nodo] = {
                'data': df,
                'padres': padres,
                'columna_valor': columna_valor,
                'columna_probabilidad': 'probabilidad',
                'archivo': archivo_cpt
            }

            print(f" CPT de {nombre_nodo} cargada - Padres: {padres}")
            return nombre_nodo

        except Exception as e:
            print(f"Error cargando CPT {archivo_cpt}: {e}")
            return None
    def obtener_probabilidad(self, nodo, evidencia):

        #Obtiene para cualquier red evidencia = diccionario {variable: valor}

        if nodo not in self.cpts:
            raise ValueError(f"No hay CPT para {nodo}")

        cpt_info = self.cpts[nodo]
        df = cpt_info['data']
        padres = cpt_info['padres']
        columna_valor = cpt_info['columna_valor']

        # Filtrar basado en la evidencia
        mascara = pd.Series([True] * len(df))

        for variable, valor in evidencia.items():
            if variable in df.columns:
                mascara &= (df[variable] == valor)

        # Encontrar filas que coincidan con la evidencia
        filas_coincidentes = df[mascara]

        if len(filas_coincidentes) == 0:
            raise ValueError(f"No se encontraron entradas para la evidencia: {evidencia}")

        # Devolver distribucion de probabilidad
        resultado = {}
        for _, fila in filas_coincidentes.iterrows():
            valor_nodo = fila[columna_valor]
            probabilidad = fila['probabilidad']
            resultado[valor_nodo] = probabilidad

        return resultado
    def mostrar_grafo(self):
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(self.grafo, k=3, iterations=100)

        nx.draw(self.grafo, pos, with_labels=True, node_color='lightcoral',
                node_size=4000, font_size=11, font_weight='bold',
                arrows=True, arrowsize=25, edge_color='gray', width=2,
                alpha=0.8)

        plt.title("Sistema de Diagnostico Medico-Red Bayesiana", size=16, pad=20)
        plt.axis('off')
        plt.show()

        print(f"Variables: {list(self.variables)}")
        print(f"Número de variables: {len(self.variables)}")
        print("\nRelaciones de dependencia:")
        for padre, hijo in self.grafo.edges():
            print(f"  {padre} → {hijo}")

        print("\nResumen por variable:")
        for variable in sorted(self.variables):
            padres = list(self.grafo.predecessors(variable))
            hijos = list(self.grafo.successors(variable))
            print(f"  {variable}:")
            print(f"    Factores de riesgo: {padres}")
            print(f"    Afecta a: {hijos}")

    def mostrar_cpts(self):
        for nodo, cpt_info in self.cpts.items():
            df = cpt_info['data']
            padres = cpt_info['padres']

            print(f"\n🔬 {nodo} | Factores: {padres}")
            print(f"Tamaño de CPT: {len(df)} combinaciones")
            if len(df) > 6:
                print("Muestra de probabilidades:")
                print(df.head(6).to_string(index=False))
            else:
                print("Probabilidades completas:")
                print(df.to_string(index=False))
            print("---")

if __name__ == "__main__":

    # Crear instancia de la red bayesiana
    rb = RedBayesiana()

    # 1. Cargar estructura
    print("\n Cargando estructura del sistema médico...")
    if not rb.cargar_estructura("estructura_red.csv"):
        print(" Error: No se pudo cargar la estructura. Verifica el archivo 'estructura_red.csv'")
        exit(1)

    # 2. Cargar las CPTs
    print("\n Cargando tablas de probabilidad médica...")
    archivos_cpt = [
        "nodo_edad.csv",
        "nodo_fumador.csv",
        "nodo_ejercicio.csv",
        "nodo_presionArterial.csv",
        "nodo_colesterol.csv",
        "nodo_enfermedadCardiaca.csv"
    ]

    cpts_cargadas = []

    for archivo in archivos_cpt:
        if os.path.exists(archivo):
            # Extraer nombre del nodo del nombre del archivo
            nombre_nodo = archivo.replace('nodo_', '').replace('.csv', '')
            nodo_cargado = rb.cargar_cpt(archivo, nombre_nodo)
            if nodo_cargado:
                cpts_cargadas.append(nodo_cargado)
        else:
            print(f" Archivo no encontrado: {archivo}")

    print(f" CPTs cargadas exitosamente: {cpts_cargadas}")

    if not cpts_cargadas:
        print(" Error: No se pudieron cargar las CPTs. Verifica los archivos:")
        for archivo in archivos_cpt:
            print(f"   - {archivo} → {' Existe' if os.path.exists(archivo) else ' No existe'}")
        exit(1)

    # 3. Mostrar información completa del sistema
    print("\n Visualizando la red de diagnóstico...")
    rb.mostrar_grafo()
    rb.mostrar_cpts()


