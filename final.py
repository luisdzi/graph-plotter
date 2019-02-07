#! /usr/bin/python

import argparse
import random
import numpy as np
import math
import Gnuplot, Gnuplot.funcutils
import time

#Defino el tipo del grafo
#Verbose es la cantidad de informacion a imprimir
class GraphData:
    def __init__(self,grafo,itera,temperature,c1,c2,verbose,verbose2):
        self.grafo=grafo #Agrego el grafo
        self.nodes=grafo[0] #El conjunto de nodos
        self.edges=grafo[1] #El conjunto de aristas
        self.itera=itera #Cantidad de iteraciones
        self.W=500 #Ancho del area de dibujo
        self.L=500 #Alto del area de dibujo
        self.c1=c1 #Fuerza de atraccion
        self.c2=c2 #Fuerza de repulsion
        self.verbose=verbose #Nivel 1 de informacion
        self.verbose2=verbose2 #Nivel 2 de informacion
        self.posiciones={} 
        self.fuerzas={}
        self.refresh=1
        self.temperature=temperature
        self.g=None #Elemento a plotear
        
    def log(self,msg):
        if (self.verbose or self.verbose2):
            print msg
        
    def log2(self,msg):
        if self.verbose2:
            print msg
        
    def f_a(self,d):
        return d**2/self.c1

    def f_r(self,d):
        return self.c2**2/d
    
    #Esta funcion asigna posiciones aleatorias a los nodos
    def randomize(self):
        self.log("Inicializando nodos...")
        to_dictionary=[] #Usaremos la lista para crear el diccionario con las posiciones
        for i in range(0,len(self.nodes)):
            pos_nodo=np.array([random.uniform(-self.W/2,self.W/2),random.uniform(-self.L/2,self.L/2)], dtype=np.float) #Posicion de cada nodo
            tempo=(self.nodes[i],pos_nodo)
            to_dictionary.append(tempo)
        self.posiciones=dict(to_dictionary)
        
    def cool(self):
        self.temperature = self.temperature * 0.95
        
    def min_f_t(self,disp):
        return min(disp,self.temperature)
        
    #Una iteracion para calcular posiciones
    def step(self):
        self.log("\nCalculando fuerzas...\nTemperatura actual: {}\n".format(self.temperature))
        N=len(self.nodes)
        
        #Las fuerzas comienzan en cero
        for i in range(0,N):
            f_node=np.empty([1,2], dtype=np.float)
            f_node=[0,0]
            self.fuerzas[self.nodes[i]]=f_node
            self.log("Posicion anterior del nodo {} en x e y: {} {}".format(self.nodes[i],
                                                                               self.posiciones[self.nodes[i]][0],
                                                                               self.posiciones[self.nodes[i]][1]))
            
        #Calculamos fuerzas de repulsion
        for i in range(N):
            node1=self.nodes[i]
            for j in range(i+1,N):
                node2=self.nodes[j]
                delta=self.posiciones[node1]-self.posiciones[node2]
                mod_delta=max(np.linalg.norm(delta),0.01)
                self.fuerzas[node1]+=((delta/mod_delta)*self.f_r(mod_delta))
                self.fuerzas[node2]-=((delta/mod_delta)*self.f_r(mod_delta))
                self.log2("Modulo fuerza de repulsion entre el nodo {} y {} en x e y: {}".format(node1,node2,mod_delta))
        
        #Calculamos las fuerzas de atraccion
        for edge in self.edges:
            delta=self.posiciones[edge[0]]-self.posiciones[edge[1]]
            mod_delta=max(np.linalg.norm(delta),0.01)
            self.fuerzas[edge[0]]-=(delta/mod_delta)*self.f_a(mod_delta)
            self.fuerzas[edge[1]]+=(delta/mod_delta)*self.f_a(mod_delta)
            self.log2("Modulo fuerza de atraccion sobre la arista {}---{}: {}".format(edge[0],edge[1],mod_delta))           
        self.log("\n")
        
        #Actualizamos las posiciones
        for node in self.nodes:
            #Primero limitamos por la temperatura
            disp=self.fuerzas[node]
            mod_disp=max(np.linalg.norm(disp),0.01)
            self.posiciones[node]+=(disp/mod_disp)*self.min_f_t(mod_disp)
            
            self.log("Posicion actualizada del nodo {} en x e y: {} {}".format(node,
                                                                               self.posiciones[node][0],
                                                                               self.posiciones[node][1]))
            
        self.cool()
 
 
    #Funcion para calcular el maximo del eje x
    def xrange(self):
        posx=[]
        for node in self.grafo[0]:
            posx.append(self.posiciones[node][0])
        resul=(max(posx),min(posx))
        return resul
    
    #Funcion para calcular el maximo del eje y
    def yrange(self):
        posy=[]
        for node in self.grafo[0]:
            posy.append(self.posiciones[node][1])
        resul=(max(posy),min(posy))
        return resul
    
    def dibujar1(self):
        xrangemin=self.xrange()[1]
        xrangemax=self.xrange()[0]
        yrangemin=self.yrange()[1]
        yrangemax=self.yrange()[0]

        #El if es para decidir si debemos replotear o no
        if self.g == None:
            self.g = Gnuplot.Gnuplot()
            self.g('set title "Grafo"')
            self.g('set xrange [{}:{}]; set yrange [{}:{}]'.format(xrangemin-100,xrangemax+100,
                                                                   yrangemin-100,yrangemax+100))
            u=1
            i=1
            for nodi in self.grafo[0]:
                self.g('set object {} circle center {},{} size 5 fc rgb "black"'.format(i,self.posiciones[nodi][0],self.posiciones[nodi][1]))
                i+=1
            for edgi in self.grafo[1]:
                posi1x=self.posiciones[edgi[0]][0]
                posi1y=self.posiciones[edgi[0]][1]
                posi2x=self.posiciones[edgi[1]][0]
                posi2y=self.posiciones[edgi[1]][1]
                self.g('set arrow {} nohead from {},{} to {},{}'.format(u,posi1x,posi1y,posi2x,posi2y))
                u+=1
            self.g('unset key')
            self.g('plot NaN')
            time.sleep(0.2)
            for x in range(1,i):
                self.g('unset object {}'.format(i))
            for x2 in range(1,u):
                self.g('unset arrow {}'.format(x2))
        else:
            #self.g('set xrange [{}:{}]; set yrange [{}:{}]'.format(-500,500,-500,500))
            self.g('set xrange [{}:{}]; set yrange [{}:{}]'.format(xrangemin-100,xrangemax+100,
                                                                   yrangemin-100,yrangemax+100))
            i=1
            u=1
            for nodi in self.grafo[0]:
                self.g('set object {} circle center {},{} size 5 fc rgb "black"'.format(i,self.posiciones[nodi][0],self.posiciones[nodi][1]))
                i+=1
            for edgi in self.grafo[1]:
                posi1x=self.posiciones[edgi[0]][0]
                posi1y=self.posiciones[edgi[0]][1]
                posi2x=self.posiciones[edgi[1]][0]
                posi2y=self.posiciones[edgi[1]][1]
                self.g('set arrow {} nohead from {},{} to {},{}'.format(u,posi1x,posi1y,posi2x,posi2y))
                u+=1
            self.g('replot')
            for x in range(1,i):
                self.g('unset object {}'.format(i))
            for x2 in range(1,u):
                self.g('unset arrow {}'.format(x2))
            time.sleep(0.1)
    
    def layout(self):
        #Empezamos en posiciones aleatorias
        self.randomize()
        
        if (self.refresh > 0):
            self.dibujar1()

        #Bucle principal
        for i in range(0, self.itera):
            #Realizar un paso de la simulacion
            self.log('Iteracion actual: {}\n'.format(i))
            self.step()
            self.log('<===============================>')
            #Si es necesario, lo mostramos por pantalla
            if (self.refresh > 0 and i % self.refresh == 0):
                #Imprimimos la iteracion actual que se muestra ploteada
                self.dibujar1()
        
        # Ultimo dibujado al final
        self.dibujar1()
        time.sleep(5)

#Para leer el grafo desde un archivo        
def leerGrafoPesoArchivo(file_path):
    V=[]
    E=[]
    with open(file_path,'r') as f: #Leemos el archivo
        #import pdb; pdb.set_trace()
        cant_nodos=int((f.readline()).strip()) #Obtengo la primera linea para saber la cantidad de nodos
        for cant_nodos in range(0,cant_nodos): #Reorremos las lienas que tienen los nodos
            V.append((f.readline()).strip()) #Agregamos los nodos
        while(True): #Leemos hasta el final del archivo
            line=(f.readline()).split() #Obtenemos los elementos para las aristas
            if (line==[]): #Si la linea es vacia, no hay nada, terminamos el ciclo
                f.close() #Cerramos el programa primero
                break
            arista=(line[0],line[1]) #Me formo la arista
            E.append(arista) #Lo agregamos al conjunto de aristas
    return (V,E)

#Si no se ingresa desde un archivo, podemos tomar el grafo por stdin directamente    
def leer_grafo_stdin():
    V = []
    E = []
    n = input()
    for i in range(n):
        v = raw_input().strip()
        V.append(v)
    while True:
        try:
            e = raw_input().split()
            e = (e[0],e[1])
            E.append(e)
        except:    
            break

    G = (V,E)
    return G

def main():
    # Definimos los argumentos de linea de comando que aceptamos
    parser = argparse.ArgumentParser()

    # Verbosidad, opcional, False por defecto
    parser.add_argument('-v', '--verbose', 
                        action='store_true', 
                        help='Muestra mas informacion - Nivel 1')
    
    # Verbosidad, opcional, False por defecto
    parser.add_argument('-vv', '--verbose2', 
                        action='store_true', 
                        help='Muestra mas informacion - Nivel 2')
    
    # Cantidad de iteraciones, opcional, 50 por defecto
    parser.add_argument('-i','--iters', 
                        type=int,
                        help='Cantidad de iteraciones a efectuar',
                        default=50)
    
    # Temperatura
    parser.add_argument('-t','--temperature',
                        type=int,
                        help='Cantidad de grados centigrados',
                        default=200)
    
    #Constante de atraccion
    parser.add_argument('-c1','--c1', 
                        type=float,
                        help='Constante de atraccion',
                        default=2000.0)
    
    #Constante de repulsion
    parser.add_argument('-c2','--c2', 
                        type=float,
                        help='Constante de repulsion',
                        default=30.0)
    
    #Archivo desde el cual leer el grafo
    parser.add_argument('-g','--grafo',
                        type=str,
                        help='Nombre del archivo con el grafo a leer')     
    
    args = parser.parse_args()
    
    if(args.grafo): #Si llame al archivo con -g
        grafo1 = leerGrafoPesoArchivo(args.grafo)
    else: #Si llame al archivo sin -g
        grafo1 = leer_grafo_stdin()
        
    ly_gr = GraphData(
        grafo1,
        args.iters, # Iteraciones
        args.temperature,
        args.c1,
        args.c2,
        verbose=args.verbose,
        verbose2=args.verbose2
    )
    
    ly_gr.layout()

    return

if __name__ == '__main__':
    main()  
