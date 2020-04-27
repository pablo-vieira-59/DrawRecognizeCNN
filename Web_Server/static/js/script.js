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
var cam_feed = document.getElementById("cam-feed");

function setOnline(div, icon){
    div.style.color = "green";
    icon.className = "fa fa-check-circle";
}

function setOffline(div, icon){
    div.style.color = "red";
    icon.className = "fa fa-times-circle";
}

function captureImage(){
    var req = $.get('/data');
    req.done( 
        function (result) {
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

cam_feed.src = "http://127.0.0.1/framed"