'use strict';

var apple = document.getElementById("apple");
var apple_icon = document.getElementById("apple-icon");
var axe = document.getElementById("axe");
var axe_icon = document.getElementById("axe-icon");
var bicycle = document.getElementById("bicycle");
var bicycle_icon = document.getElementById("bicycle-icon");
var book = document.getElementById("book");
var book_icon = document.getElementById("book-icon");
var bus = document.getElementById("bus");
var bus_icon = document.getElementById("bus-icon");

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const snap = document.getElementById("snap");


const constraints = {
    audio: false,
    video:{
        width: 640, height: 480
    }
};

async function init(){
    try{
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        handleSuccess(stream);
    }
    catch(e){
        console.log(e);
    }
}

function handleSuccess(stream){
    window.stream = stream;
    video.srcObject = stream;
}

function setOnline(div, icon){
    div.style.color = "green";
    icon.className = "fa fa-check-circle";
}

function setOffline(div, icon){
    div.style.color = "red";
    icon.className = "fa fa-times-circle";
}

function captureImage(){
    context.drawImage(video, 0,0, 640, 480);
    var imgURL = canvas.toDataURL();
    $.ajax({
        type: "POST",
        url: "/load_camera",
        data: {
            image:imgURL
        },
        success: function(data) {
          return;
        },
        error: function(data) {
          alert('There was an error uploading your file!');
        }
      }).done(function (result) {
        setOffline(apple, apple_icon);
        setOffline(axe, axe_icon);
        setOffline(bicycle, bicycle_icon);
        setOffline(book, book_icon);
        setOffline(bus, bus_icon);

        if(result.class == 0){
            setOnline(apple, apple_icon);
        }

        if(result.class == 1){
            setOnline(axe, axe_icon);
        }

        if(result.class == 2){
            setOnline(bicycle, bicycle_icon);
        }

        if(result.class == 3){
            setOnline(book, book_icon);
        }

        if(result.class == 4){
            setOnline(bus, bus_icon);
        }

    });
}

function drawImge(){
    context.drawImage(video, 0, 0, 640, 480);

    var faceArea = 300;
    var pX=canvas.width/2 - faceArea/2;
    var pY=canvas.height/2 - faceArea/2;

    context.rect(pX,pY,faceArea,faceArea);
    context.lineWidth = "4";
    context.strokeStyle = "red";    
    context.stroke();

    setTimeout(drawImge , 20);
}

init();
var context = canvas.getContext('2d');
video.hidden = true;
video.onplay = function() {
    setTimeout(drawImge , 20);
};
