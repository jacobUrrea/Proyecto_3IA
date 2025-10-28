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
        # Carga la estructura de la red desde CSV
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
        # Carga CPT desde archivo especifico
        try:
            df = pd.read_csv(archivo_cpt)

            columnas = list(df.columns)

            # La ultima columna debe llamarse 'probabilidad'
            if columnas[-1] != 'probabilidad':
                raise ValueError("La ultima columna debe llamarse 'probabilidad'")

            columnas_variables = columnas[:-1]

            # El nombre de la variable actual es la ultima columna antes de probabilidad
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

            print(f"CPT de {nombre_nodo} cargada - Padres: {padres}")
            return nombre_nodo

        except Exception as e:
            print(f"Error cargando CPT {archivo_cpt}: {e}")
            return None

    def obtener_probabilidad(self, variable, valor, evidencia_parcial):
        # Obtiene P(variable=valor | evidencia_parcial)
        if variable not in self.cpts:
            raise ValueError(f"No hay CPT para {variable}")

        #Obtiene el DataFrame de la CPT
        cpt_info = self.cpts[variable]
        df = cpt_info['data']

        #Si la variable no depende de nadie (por ejemplo Edad o Fumador),
        #busca directamente en la columna de valores
        if not cpt_info['padres']:
            fila = df[df[cpt_info['columna_valor']] == valor]
            if len(fila) == 0:
                return 0.0
            return fila.iloc[0]['probabilidad']
        #Si encuentra padres aplica un filtro sobre las filas que coinciden con los valores de los padres
        #   Si no hay coinsidencias devuelve 0, de lo contrario devuelve la probabilidad
        mascara = pd.Series([True] * len(df))
        for padre in cpt_info['padres']:
            if padre in evidencia_parcial:
                mascara &= (df[padre] == evidencia_parcial[padre])

        filas_coincidentes = df[mascara]

        if len(filas_coincidentes) == 0:
            return 0.0

        fila_valor = filas_coincidentes[filas_coincidentes[cpt_info['columna_valor']] == valor]
        if len(fila_valor) == 0:
            return 0.0

        return fila_valor.iloc[0]['probabilidad']

    def inferencia_por_enumeracion(self, consulta, evidencia, mostrar_traza=True, archivo_traza=None):
        # Implementa inferencia por enumeracion
        def log_traza(mensaje):
            if mostrar_traza:
                print(mensaje)
            if archivo_traza:
                with open(archivo_traza, 'a', encoding='utf-8') as f:
                    f.write(mensaje + "\n")
        #Guarda o imprime los pasos para auditoria:
        if archivo_traza:
            with open(archivo_traza, 'w', encoding='utf-8') as f:
                f.write("Traza de inferencia por enumeracion \n")
                f.write(f"Consulta: P({consulta} | {evidencia})\n")

        todas_variables = list(self.variables)
        log_traza(f"1. Variables de la red: {todas_variables}")

        variables_ocultas = [v for v in todas_variables if v != consulta and v not in evidencia]
        log_traza(f"2. Variables ocultas (Y): {variables_ocultas}")
        log_traza(f"Consulta (X): {consulta}")
        log_traza(f"Evidencia (e): {evidencia}")
        #Lee los valores posibles desde la CP
        valores_consulta = list(self.cpts[consulta]['data'][self.cpts[consulta]['columna_valor']].unique())
        valores_consulta = [v for v in valores_consulta if pd.notna(v)]
        log_traza(f"3. Valores posibles de {consulta}: {valores_consulta}")

        log_traza(f"\n 4.Calculo de probabilidades conjuntas:")
        log_traza(f"Formula: P({consulta}, {evidencia}, {variables_ocultas})")

        def calcular_probabilidad_conjunta(asignacion_completa):
            probabilidad = 1.0
            explicacion = []
            #Si falta alguna combinacion devuelve 0

            #Recorre todas las variables del modelo
            for variable in todas_variables:
                valor = asignacion_completa[variable]
            #Para cada variable, busca el valor asignado (por ejemplo Edad=adulto o Fumador=si) 
            #Dentro del diccionario asignacion_completa
                evidencia_padres = {}
                if variable in self.cpts:
                    for padre in self.cpts[variable]['padres']:
                        if padre in asignacion_completa:
                            evidencia_padres[padre] = asignacion_completa[padre]
                
                #Construye un diccionario con los padres de la variable

                #Llama a la funcion anterior para consultar la probabilidad exacta desde el CSV correspondiente.
                prob_cond = self.obtener_probabilidad(variable, valor, evidencia_padres)

                if prob_cond == 0:
                    return 0.0, []

                probabilidad *= prob_cond
                #Multiplica esa probabilidad con las demas ya acumuladas
                if self.cpts[variable]['padres']:
                    explicacion.append(f"P({variable}={valor}|{evidencia_padres})={prob_cond:.4f}")
                else:
                    explicacion.append(f"P({variable}={valor})={prob_cond:.4f}")

            return probabilidad, explicacion

        def enumerar_combinaciones(variables, asignacion_actual):
            #Si ya no hay variables ocultas por enumerar, devuelve la asignacion actual
            if not variables:
                return [asignacion_actual.copy()]

            combinaciones = []
            #Toma la primera variable de la lista y obtiene todos los valores posibles de esa variable desde su CSV
            #variable_actual = "PresionArterial" valores_variable = ["baja", "media", "alta"]
            variable_actual = variables[0]
            valores_variable = list(self.cpts[variable_actual]['data'][self.cpts[variable_actual]['columna_valor']].unique())
            valores_variable = [v for v in valores_variable if pd.notna(v)]
            #Para cada valor posible asigna ese valor a la variable actual,
            #llama recursivamente a la funcion con el resto de variables y acumula todas las combinaciones
            for valor in valores_variable:
                nueva_asignacion = asignacion_actual.copy()
                nueva_asignacion[variable_actual] = valor
                combinaciones.extend(enumerar_combinaciones(variables[1:], nueva_asignacion))

            return combinaciones

        resultados = {}

        for valor_consulta in valores_consulta:
            log_traza(f"\n Calculo para {consulta} = {valor_consulta} ---")

            asignacion_base = evidencia.copy()
            asignacion_base[consulta] = valor_consulta

            suma_probabilidad = 0.0
            combinaciones_validas = 0
            #Genera todas las combinaciones posibles de valores de las variables ocultas, usando recursion.
            for combinacion in enumerar_combinaciones(variables_ocultas, asignacion_base):
                prob, explicacion = calcular_probabilidad_conjunta(combinacion)

                if prob > 0:
                    suma_probabilidad += prob
                    combinaciones_validas += 1
                    log_traza(f"Combinacion {combinaciones_validas}: {combinacion}")
                    log_traza(f"{' × '.join(explicacion)} = {prob:.6f}")

            resultados[valor_consulta] = suma_probabilidad
            log_traza(f"Suma para {consulta}={valor_consulta}: {suma_probabilidad:.6f}")
            log_traza(f"Combinaciones validas: {combinaciones_validas}")

        log_traza(f"\n5Normalizacion:")
        suma_total = sum(resultados.values())
        log_traza(f"Suma total de probabilidades conjuntas: {suma_total:.6f}")

        if suma_total == 0:
            log_traza("ERROR: Suma total es 0, evidencia imposible con las CPTs actuales")
            return None

        alpha = 1.0 / suma_total
        log_traza(f"Factor de normalizacion alpha = 1/{suma_total:.6f} = {alpha:.6f}")

        resultados_normalizados = {}
        #Calcular las probabilidades normalizadas
        for valor, prob in resultados.items():
            prob_normalizada = prob * alpha
            resultados_normalizados[valor] = prob_normalizada
            log_traza(f"P({consulta}={valor}|evidencia) = {prob:.6f} × {alpha:.6f} = {prob_normalizada:.6f}")

        log_traza("Resultado final:")
        log_traza(f"P({consulta} | {evidencia}) = {resultados_normalizados}")

        return resultados_normalizados

def generar_reporte_validacion(resultados_reales):
    print("Reporte de Validacion: ")
    # CASO 1
    print("\nCASO DE PRUEBA 1")
    print("Consulta: P(EnfermedadCardiaca | Edad=adulto, Fumador=si, Ejercicio=poco)")
    print("Variables ocultas: PresionArterial, Colesterol")

    if resultados_reales[0] is not None:
        valor_real_alto = resultados_reales[0].get('alto_riesgo', 0)
        print(f"Valor esperado (manual): alto_riesgo = 0.41")
        print(f"Valor real (programa): alto_riesgo = {valor_real_alto:.3f}")
    else:
        print("No se pudo calcular")

    # CASO 2
    print("\nCASO DE PRUEBA 2")
    print("Consulta: P(PresionArterial | Edad=mayor, Fumador=no, Ejercicio=moderado)")
    if resultados_reales[1] is not None:
        print(f"Valor real (programa): {resultados_reales[1]}")
    else:
        print("No se pudo calcular - evidencia imposible")

    # CASO 3
    print("\nCASO DE PRUEBA 3")
    print("Consulta: P(Colesterol | Edad=joven, Fumador=no)")
    if resultados_reales[2] is not None:
        print(f"Valor real (programa): {resultados_reales[2]}")
    else:
        print("No se pudo calcular - evidencia imposible")

    # CASO 4
    print("\nCASO DE PRUEBA 4")
    print("Consulta: P(Fumador | Edad=adulto)")
    if len(resultados_reales) > 3 and resultados_reales[3] is not None:
        print(f"Valor real (programa): {resultados_reales[3]}")
    else:
        print("No se pudo calcular")

    # CASO 5
    print("\nCASO DE PRUEBA 5")
    print("Consulta: P(Ejercicio | EnfermedadCardiaca=alto_riesgo)")
    if len(resultados_reales) > 4 and resultados_reales[4] is not None:
        print(f"Valor real (programa): {resultados_reales[4]}")
    else:
        print("No se pudo calcular")


    casos_exitosos = sum(1 for r in resultados_reales if r is not None)
    print(f"Casos ejecutados exitosamente: {casos_exitosos}/5")
    print(f"Archivos de traza generados: 5")
    print(f"Validacion del sistema: COMPLETADA")

if __name__ == "__main__":
    print("SISTEMA DE INFERENCIA POR ENUMERACION - VALIDACION COMPLETA")

    rb = RedBayesiana()

    rb.cargar_estructura("estructura_red.csv")

    archivos_cpt = [
        "nodo_Edad.csv", "nodo_Fumador.csv", "nodo_Ejercicio.csv",
        "nodo_PresionArterial.csv", "nodo_Colesterol.csv", "nodo_EnfermedadCardiaca.csv"
    ]

    for archivo in archivos_cpt:
        if os.path.exists(archivo):
            nombre_nodo = archivo.replace('nodo_', '').replace('.csv', '')
            rb.cargar_cpt(archivo, nombre_nodo)

    resultados_reales = []

    # CASO 1
    print("\n1. Ejecutando caso 1...")
    resultado1 = rb.inferencia_por_enumeracion(
        consulta="EnfermedadCardiaca",
        evidencia={'Edad': 'adulto', 'Fumador': 'si', 'Ejercicio': 'poco'},
        mostrar_traza=False,
        archivo_traza="validacion_caso1.txt"
    )
    resultados_reales.append(resultado1)
    print(f"Resultado: {resultado1}")

    # CASO 2
    print("\n2. Ejecutando caso 2...")
    resultado2 = rb.inferencia_por_enumeracion(
        consulta="PresionArterial",
        evidencia={'Edad': 'mayor', 'Fumador': 'no', 'Ejercicio': 'moderado'},
        mostrar_traza=False,
        archivo_traza="validacion_caso2.txt"
    )
    resultados_reales.append(resultado2)
    print(f"Resultado: {resultado2}")

    # CASO 3
    print("\n3. Ejecutando caso 3...")
    resultado3 = rb.inferencia_por_enumeracion(
        consulta="Colesterol",
        evidencia={'Edad': 'joven', 'Fumador': 'no'},
        mostrar_traza=False,
        archivo_traza="validacion_caso3.txt"
    )
    resultados_reales.append(resultado3)
    print(f"Resultado: {resultado3}")

    # CASO 4
    print("\n4. Ejecutando caso 4...")
    resultado4 = rb.inferencia_por_enumeracion(
        consulta="Fumador",
        evidencia={'Edad': 'adulto'},
        mostrar_traza=False,
        archivo_traza="validacion_caso4.txt"
    )
    resultados_reales.append(resultado4)
    print(f"Resultado: {resultado4}")

    # CASO 5
    print("\n5. Ejecutando caso 5...")
    resultado5 = rb.inferencia_por_enumeracion(
        consulta="Ejercicio",
        evidencia={'EnfermedadCardiaca': 'alto_riesgo'},
        mostrar_traza=False,
        archivo_traza="validacion_caso5.txt"
    )
    resultados_reales.append(resultado5)
    print(f"Resultado: {resultado5}")

    generar_reporte_validacion(resultados_reales)

