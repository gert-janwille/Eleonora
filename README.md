<div align="center">
  <a href="https://github.com/gert-janwille/Eleonora">
    <img width="300"" src="https://raw.github.com/gert-janwille/Eleonora/master/docs/assets/eleonora-official.png">
  </a>
  <br/>
  <br/>
  <br/>
  <p>
    Mrs. Eleonora is an Artificial Intelligence Assistant Robot made to recognise, greet and learn people. She will recognise emotions to be able to interact with the user and give the user support.
</div>


![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Gert-Jan Wille](https://img.shields.io/badge/Author-gert--janwille-blue.svg)



## Getting Started

Eleonora is build to run on Mac OSX or a Rasberry Pi. To run the code just clone the folder to your machine and cd into the folder.
Run the following command `$ python bin/eleonora` and she'll be up and running.

[!] Because of the file size of the trained models, you can download them in folowing link: [DOWNLOAD](https://drive.google.com/open?id=1c7h-AqmnC-DoYonKh4CPXtDpLlOe2Aff) and add them to the data/models folder.


## Installation

Install all needed packages
```
$ pip install -r requirements.txt
```

Download the trained models from the [Google Drive](https://drive.google.com/open?id=1c7h-AqmnC-DoYonKh4CPXtDpLlOe2Aff) and place them in the folder faces.

Run the following command to start Eleonora
```
$ python bin/eleonora
```

### Training
If you want to train the models run the following command to execute the trainings scripts.
```
$ python bin/eleonora --Train
```

<br/>

### Interaction Functions
* **Mindfulness** - This will make the heart rate go low so the user will reduce his/her stress level
* **Hugging** - Eleonora can't hug of her own. But she can ask to give her a hug.
* **Joking** - She can be very funny and will tell you a joke when you need it.
* ...

<br/>

## Built With

* [TensorFlow](https://www.tensorflow.org/) - Artificial Intelligence Framework
* [OpenCV](https://opencv.org/) - Face Detection
* [Snowboy](https://snowboy.kitt.ai) - Hotword Detection

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/gert-janwille/Eleonora/tags).

## Authors

* **Gert-Jan Wille** - *Initial work* - [gert-janwille](https://github.com/gert-janwille)

See also the list of [contributors](https://github.com/gert-janwille/Eleonora/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* Accomplished by [NMCT](http://www.howest.be/Default.aspx?target=pih&lan=nl&item=71&gclid=EAIaIQobChMI29Cbq9-41wIVA2wbCh20MwlUEAAYASAAEgIwYPD_BwE) & [Devine](http://www.howest.be/Default.aspx?target=pih&lan=nl&item=1094)
