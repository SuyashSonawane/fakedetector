document.querySelector('#imageUpload').addEventListener('change', event => {
    handleImageUpload(event)
})
document.querySelector('#videoUpload').addEventListener('change', event => {
    handleVideoUpload(event)
})

// const API = "https://cdacfakeapidocker.azurewebsites.net/"
const API = "http://localhost:3000/"

const handleImageUpload = event => {
    const files = event.target.files
    const formData = new FormData()
    formData.append('File', files[0])
    if (files[0].size > 5000000) {
        alert("File too big")
        document.querySelector('.img').style.display = 'none'
        return
    }

    // console.log(formData)
    document.querySelector(".img").style.display = 'block'

    fetch(API + 'uploadI', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            // console.log(data)
            document.querySelector('.img').style.display = 'none'
            alert(data.result)
            // if (data.result > 0.5)
            //     alert('fake')
            // else
            //     alert('real')
        })
        .catch(error => {
            // console.error(error)
            alert(error)
        })
}
const handleVideoUpload = event => {
    const files = event.target.files
    const formData = new FormData()
    formData.append('File', files[0])

    // console.log(formData)
    document.querySelector(".img").style.display = 'block'

    if (files[0].size > 5000000) {
        alert("File too big")
        document.querySelector('.img').style.display = 'none'
        return
    }

    fetch(API + 'uploadV', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            // console.log(data)
            document.querySelector('.img').style.display = 'none'
            alert(data.result)
            // if (data.result > 0.5)
            //     alert('fake')
            // else
            //     alert('real')
        })
        .catch(error => {
            // console.error(error)
            alert(error)
        })
}

function sendLink() {
    let url = document.getElementById("link").value
    if (url) {
        document.querySelector(".img").style.display = 'block'
        fetch(API + 'link', {
            method: 'POST',
            body: JSON.stringify({ 'link': url }),
            headers: {
                "Content-Type": "application/json"
            }
        })
            .then(response => response.json())
            .then(data => {
                console.log(data)
                document.querySelector('.img').style.display = 'none'
                alert(data.result)
                // if (data.result > 0.5)
                //     alert('fake')
                // else
                //     alert('real')
            })
            .catch(error => {
                // console.error(error)
                alert(error)
            })
    }
}