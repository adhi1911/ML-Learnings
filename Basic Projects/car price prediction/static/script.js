






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






 

