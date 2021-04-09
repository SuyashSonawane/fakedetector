document.querySelector('#imageUpload').addEventListener('change', event => {
    handleImageUpload(event)
})
document.querySelector('#videoUpload').addEventListener('change', event => {
    handleVideoUpload(event)
})

const handleImageUpload = event => {
    const files = event.target.files
    const formData = new FormData()
    formData.append('File', files[0])

    console.log(formData)
    document.querySelector(".img").style.display = 'block'

    fetch('http://localhost:3000/uploadI', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            console.log(data)
            document.querySelector('.img').style.display = 'none'
            if (data.result > 0.5)
                alert('fake')
            else
                alert('real')
        })
        .catch(error => {
            console.error(error)
            alert(error)
        })
}
const handleVideoUpload = event => {
    const files = event.target.files
    const formData = new FormData()
    formData.append('File', files[0])

    console.log(formData)
    document.querySelector(".img").style.display = 'block'

    fetch('http://localhost:3000/uploadV', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            console.log(data)
            document.querySelector('.img').style.display = 'none'
            if (data.result > 0.5)
                alert('fake')
            else
                alert('real')
        })
        .catch(error => {
            console.error(error)
            alert(error)
        })
}