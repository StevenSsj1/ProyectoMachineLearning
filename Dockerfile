# Utiliza la imagen oficial de Redis desde Docker Hub
FROM redis:latest

# Expone el puerto 6379
EXPOSE 6379

# Comando por defecto para ejecutar Redis
CMD ["redis-server"]
