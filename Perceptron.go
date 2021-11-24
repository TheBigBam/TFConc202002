//sudo lsof -i -P -n
//sudo fuser -k Port_Number/tcp
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math/rand"
	"net"
	"net/http"
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

func targetPredict(targets []int, wanted int) []int {
	nTarget := make([]int, len(targets))
	for i := 0; i < len(targets); i++ {
		if targets[i] == wanted {
			nTarget[i] = -1
		} else {
			nTarget[i] = 1
		}
	}
	return nTarget
}

type Data struct {
	SepalL float64 `json:"sepal_length"`
	SepalW float64 `json:"sepal_width"`
	PetalL float64 `json:"petal_length"`
	PetalW float64 `json:"petal_width"`
	Class  string  `json:"class"`
}

func readJSON() ([]Data, error) {
	url := "https://raw.githubusercontent.com/diananr/concurrente-202102/main/backend/dataset/irisJson.json"
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}

	defer resp.Body.Close()
	var arrIrisT []Data
	buf := new(bytes.Buffer)
	buf.ReadFrom(resp.Body)
	respByte := buf.Bytes()
	body := bytes.TrimPrefix(respByte, []byte("\xef\xbb\xbf"))

	if err := json.Unmarshal([]byte(body), &arrIrisT); err != nil {
		return nil, err
	}
	return arrIrisT, nil
}

func SplitData(data []Data) ([][]float64, []int) {
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(data), func(i, j int) { data[i], data[j] = data[j], data[i] })
	fmt.Println(data[0].Class)
	x := make([][]float64, len(data))
	y := []int{}
	keys := make(map[string]bool)
	list := []string{}

	for i, entry := range data {
		if _, value := keys[entry.Class]; !value {
			keys[entry.Class] = true
			list = append(list, entry.Class)
			for i, irisType := range list {
				if entry.Class == irisType {
					y = append(y, i)
				}
			}
		} else {
			for i, irisType := range list {
				if entry.Class == irisType {
					y = append(y, i)
				}
			}
		}

		xRow := make([]float64, 4)
		for j := 0; j < 4; j++ {
			listOfColumns := [4]float64{data[i].PetalL, data[i].PetalW, data[i].SepalL, data[i].SepalW}
			xRow[j] = listOfColumns[j]
		}
		x[i] = xRow
	}
	return x, y
}

func server(hostname string, end chan bool, neuronFinal *Perceptron) {
	ln, _ := net.Listen("tcp", hostname)
	defer ln.Close()
	fmt.Println("Listening!")
	for {
		con, _ := ln.Accept()
		handle(con, hostname, end, neuronFinal)
	}
}

func handle(con net.Conn, hostname string, end chan bool, neuronFinal *Perceptron) {
	defer con.Close()
	dec := json.NewDecoder(con)
	var msg Msg
	print(con)
	if err := dec.Decode(&msg); err == nil {
		fmt.Printf("Message: %v\n", msg)

		if msg.Option == "msg" {
			sendPerceptronEntrenar(hostname, msg.Addr)
		}
		if msg.Option == "per" {
			perAux := stringToArray(msg.Message)
			neuronFinal.sumarPesos(perAux)
			end <- true
		}
	} else {
		fmt.Println("ErrorH: ", err)
	}
	//end <- true
}

type Msg struct {
	Addr    string `json:"addr"`
	Option  string `json:"option"`
	Message string `json:"message"`
}

func send(local, remote string, msg string) {
	if remote != "0" {
		con, _ := net.Dial("tcp", remote)
		defer con.Close()
		enc := json.NewEncoder(con)
		if err := enc.Encode(Msg{local, "msg", msg}); err == nil {
			fmt.Printf("Sending %s to %s\n", msg, remote)
		} else {
			fmt.Println("ErrorS: ", err)
		}
	}
}

func sendPerceptronEntrenar(local, remote string) {
	data, _ := readJSON()
	x, y := SplitData(data)
	y = targetPredict(y, 0)
	neuron := Perceptron{rate: 0.1, iterNum: 50}
	neuron.Fit(x, y, 4)
	fmt.Println("Resultado:", neuron.Resultado(x[0]), "\tTrue: ", y[0])
	fmt.Println("Accuracy: ", neuron.Accuracy(x, y))

	if remote != "0" {
		con, _ := net.Dial("tcp", remote)
		defer con.Close()
		enc := json.NewEncoder(con)
		if err := enc.Encode(Msg{local, "per", arrayToString(neuron.getPesos())}); err == nil {
			fmt.Print("Sending \n", neuron.getPesos(), " to ", remote)
		} else {
			fmt.Println("ErrorSend: ", err)
		}
	}
}

func arrayToString(array []float64) string {
	newArray := strconv.FormatFloat(array[0], 'f', 6, 64)
	for i := 1; i < len(array); i++ {
		newArray = newArray + "," + strconv.FormatFloat(array[i], 'f', 6, 64)
	}
	return newArray
}

func stringToArray(text string) []float64 {
	stringsArray := strings.Split(text, ",")
	var newArray []float64
	for _, element := range stringsArray {
		floatAux, _ := strconv.ParseFloat(element, 64)
		newArray = append(newArray, floatAux)
	}
	return newArray
}

var nWaits = 0

func main() {

	var hostname string
	var remote []string
	var test string
	fmt.Print("Hostname: ")
	fmt.Scanf("%s", &hostname)
	fmt.Scanf("%s", &test)
	fmt.Print(test)

	neuronFinal := Perceptron{rate: 0.1, iterNum: 50}
	neuronFinal.iniciarPesos()
	data, _ := readJSON()
	x, y := SplitData(data)
	y = targetPredict(y, 0)

	if hostname == "0" {
		hostname = fmt.Sprintf("localhost:800%s", hostname)
		var nConnections int
		fmt.Print("Numero de distribuciones: ")
		fmt.Scanf("%d", &nConnections)
		for i := 1; i <= nConnections; i++ {
			remoteAux := fmt.Sprintf("localhost:800%s", strconv.Itoa(i))
			remote = append(remote, remoteAux)
			fmt.Print(remote)
		}
		end := make(chan bool, nConnections)
		go server(hostname, end, &neuronFinal)

		for _, port := range remote {
			send(hostname, port, "hola")
		}

		for i := 0; i < nConnections; i++ {
			<-end
		}

		fmt.Println("Resultado:", neuronFinal.Resultado(x[0]), "\tTrue: ", y[0])
		fmt.Println("Accuracy: ", neuronFinal.Accuracy(x, y))
	} else if hostname > "0" {
		hostname = fmt.Sprintf("localhost:800%s", hostname)

		end := make(chan bool, 1)
		go server(hostname, end, &neuronFinal)
		<-end
	}
}
