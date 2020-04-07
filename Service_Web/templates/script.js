var cam = document.getElementById("camera");
var sns = document.getElementById("sensor");
var bat = document.getElementById("bateria");
var mic = document.getElementById("microfone");
var snd = document.getElementById("som");
Console.log(cam);

function makeReq(){
    console.log("Called")
    var req = $.get('/data');
    req.done( 
        function (result) {
            var obj = JSON.parse(result)
            cam.innerHTML = result;
            sns.innerHTML = result;
            bat.innerHTML = result;
            mic.innerHTML = result;
            snd.innerHTML = result;
        }
    );
    setTimeout(makeReq,2000);
}

//makeReq();