import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

export default function ModelViewer({ url }) {
  const containerRef = useRef(null)

  useEffect(() => {
    if (!url || !containerRef.current) return
    const container = containerRef.current

    // Scene
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x1a1a1f)

    // Camera
    const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 5000)
    camera.position.set(0, -300, 200)

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(window.devicePixelRatio)
    renderer.setSize(container.clientWidth, container.clientHeight)
    renderer.shadowMap.enabled = true
    container.appendChild(renderer.domElement)

    // Lights
    const ambient = new THREE.AmbientLight(0xffffff, 0.5)
    scene.add(ambient)

    const key = new THREE.DirectionalLight(0xffffff, 1.2)
    key.position.set(200, -200, 400)
    scene.add(key)

    const fill = new THREE.DirectionalLight(0xa0a0ff, 0.5)
    fill.position.set(-200, 200, 100)
    scene.add(fill)

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.target.set(0, 0, 0)

    // Load STL
    const loader = new STLLoader()
    loader.load(
      url,
      (geometry) => {
        geometry.computeVertexNormals()

        // Centre & fit
        geometry.computeBoundingBox()
        const box = geometry.boundingBox
        const centre = new THREE.Vector3()
        box.getCenter(centre)
        geometry.translate(-centre.x, -centre.y, -centre.z)

        const size = new THREE.Vector3()
        box.getSize(size)
        const maxDim = Math.max(size.x, size.y, size.z)
        const scaleFactor = 300 / maxDim
        geometry.scale(scaleFactor, scaleFactor, scaleFactor)

        const material = new THREE.MeshStandardMaterial({
          color: 0x6c63ff,
          roughness: 0.4,
          metalness: 0.1,
        })
        const mesh = new THREE.Mesh(geometry, material)
        scene.add(mesh)

        // Fit camera
        const sphere = new THREE.Sphere()
        geometry.computeBoundingSphere()
        sphere.copy(geometry.boundingSphere)
        const dist = sphere.radius * 2.5
        camera.position.set(0, -dist, dist * 0.8)
        controls.target.set(0, 0, 0)
        controls.update()
      },
      undefined,
      (err) => console.error('STL load error', err),
    )

    // Render loop
    let animId
    const animate = () => {
      animId = requestAnimationFrame(animate)
      controls.update()
      renderer.render(scene, camera)
    }
    animate()

    // Resize
    const onResize = () => {
      if (!container) return
      camera.aspect = container.clientWidth / container.clientHeight
      camera.updateProjectionMatrix()
      renderer.setSize(container.clientWidth, container.clientHeight)
    }
    window.addEventListener('resize', onResize)

    return () => {
      cancelAnimationFrame(animId)
      window.removeEventListener('resize', onResize)
      controls.dispose()
      renderer.dispose()
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement)
      }
    }
  }, [url])

  return (
    <div className="viewer-container" ref={containerRef}>
      <span className="viewer-hint">Drag to rotate · Scroll to zoom</span>
    </div>
  )
}
