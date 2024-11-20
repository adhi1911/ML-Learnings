

function load_car_model(companyId, carModelId) {
    let companySelect = document.getElementById(companyId);
    let carModelSelect = document.getElementById(carModelId);

    fetch('/get_car_models', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `company=${encodeURIComponent(companySelect.value)}`,
    })
    .then(response => response.text())
    .then(data => {
        carModelSelect.innerHTML = data;
    });
}


function formHandeler(event) {
    event.preventDefault();
}

function sendData() {
    let form = document.querySelector('form');
    form.addEventListener('submit', formHandeler);
    
     
        let formData = new FormData(form);
        console.log('clicked');
        console.log(formData);

        new Promise((resolve, reject) => {
            fetch('/predict', {
                method: 'POST',
                body: formData,
            })
            .then(response => response.text())
            .then(data => resolve(data))
            .catch(error => reject(error));
        })
        .then(data => {
            let result = document.getElementById('result');
            result.innerHTML = `<h3>Price of the car is: ${data}<h3>`;
        })
        .catch(error => console.error('Error:', error));
    ;
}



 


    

