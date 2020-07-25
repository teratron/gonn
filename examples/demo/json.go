package main

import (
	"encoding/json"
	"io/ioutil"
	"log"
)

type Client struct {
	ClientId string
	Date     string
}

type Settings struct {
	Clients []Client
}

const settingsFilename = "settings.json"

func main() {
	rawDataIn, err := ioutil.ReadFile(settingsFilename)
	if err != nil {
		log.Fatal("Cannot load settings:", err)
	}

	var settings Settings
	err = json.Unmarshal(rawDataIn, &settings)
	if err != nil {
		log.Fatal("Invalid settings format:", err)
	}

	newClient := Client{
		ClientId: "123",
		Date:     "2016-11-17 12:34",
	}

	settings.Clients = append(settings.Clients, newClient)

	rawDataOut, err := json.MarshalIndent(&settings, "", "\t")
	if err != nil {
		log.Fatal("JSON marshaling failed:", err)
	}

	err = ioutil.WriteFile("settings1.json", rawDataOut, 0)
	if err != nil {
		log.Fatal("Cannot write updated settings file:", err)
	}
}
