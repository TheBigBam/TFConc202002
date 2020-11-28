//sudo lsof -i -P -n
//sudo fuser -k Port_Number/tcp
package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math/rand"
	"net"
	"strconv"
	"strings"
	"time"
)

type Perceptron struct {
	rate    float64   `json:"rate"`
	iterNum int       `json:"nInter"`
	pesos   []float64 `json:"pesos"`
	errors  []int     `json:"errors"`
}

func (p *Perceptron) perceptronConc(x1 [][]float64, y1 []int, channel chan []float64) {
	auxWeight := make([]float64, len(x1[0])+1)
	for i := 0; i < p.iterNum; i++ {
		errors := 0
		for j := 0; j < len(x1); j++ {
			update := p.rate * float64(y1[j]-p.intPredict(x1[j], auxWeight))
			auxWeight[0] += update
			for k := 1; k < len(auxWeight); k++ {
				auxWeight[k] += update * x1[j][k-1]
			}
			if update != 0.0 {
				errors += 1
			}
		}

		p.errors = append(p.errors, errors)
	}
	channel <- auxWeight
}
func (p *Perceptron) Fit(x [][]float64, y []int, nThreads int) {
	auxWeight := make([]float64, len(x[0])+1)
	p.pesos = append(p.pesos, auxWeight...)
	subSetLen := int(len(x) / nThreads)
	chans := make([]chan []float64, nThreads)
	for i := range chans {
		chans[i] = make(chan []float64)
	}
	for i := 0; i < nThreads; i++ {
		go p.perceptronConc(x[i*subSetLen:(i+1)*subSetLen], y[i*subSetLen:(i+1)*subSetLen], chans[i])
	}
	for i := 0; i < nThreads; i++ {
		subWeights := <-chans[i]
		for j, pesosN := range subWeights {
			p.pesos[j] += pesosN
		}
	}
}
func (p *Perceptron) intPredict(x []float64, pesos []float64) int {
	if p.intNetIn(x, pesos) >= 0.0 {
		return 1
	}
	return -1
}

func (p *Perceptron) intNetIn(x []float64, pesos []float64) float64 {
	z := 0.0
	for i := 0; i < len(x); i++ {
		z += x[i] * pesos[i+1]
	}
	z += pesos[0]
	return z
}

func (p *Perceptron) Resultado(x []float64) int {
	if p.intNet(x) >= 0.0 {
		return 1
	}
	return -1
}

func (p *Perceptron) intNet(x []float64) float64 {
	z := 0.0
	for i := 0; i < len(x); i++ {
		z += x[i] * p.pesos[i+1]
	}
	z += p.pesos[0]
	return z
}

func (p *Perceptron) Accuracy(xT [][]float64, yT []int) float64 {
	correctPredict := 0.0
	for i := 0; i < len(xT); i++ {
		if p.Resultado(xT[i]) == yT[i] {
			correctPredict++
		}
	}
	return correctPredict / float64(len(xT))
}

func (p *Perceptron) iniciarPesos() {
	auxWeight := make([]float64, 5)
	p.pesos = append(p.pesos, auxWeight...)
}

func (p *Perceptron) getPesos() []float64 {
	return p.pesos
}

func (p *Perceptron) sumarPesos(pesosN []float64) {
	for i, _ := range p.pesos {
		p.pesos[i] += pesosN[i]
	}
}

func (p *Perceptron) dividePesos(divider int) {
	for _, element := range p.pesos {
		element = element / (float64)(divider)
	}
}
