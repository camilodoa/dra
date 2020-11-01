// border color
let now = new Date().getHours();
if (now > 20 || now < 7) {
  document.body.style.backgroundColor = '#ff9f99';
  document.getElementById("footer").style.backgroundColor = '#ff9f99';
}
else {
  document.body.style.backgroundColor = '#ffe7e5';
  document.getElementById("footer").style.backgroundColor = '#ffe7e5';
}
