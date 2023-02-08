#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup

URL = "https://realpython.github.io/fake-jobs/"
page = requests.get(URL)
soup = BeautifulSoup(page.content, "html.parser")


# In[ ]:


page.content


# In[4]:


results = soup.find(id='ResultsContainer')

print(results.prettify())


# In[12]:


job_elements = results.find_all("div", class_="card-content")
job_elements[0]


# In[13]:


for job_element in job_elements:
    title_element = job_element.find("h2", class_="title")
    company_element = job_element.find("h3", class_="company")
    location_element = job_element.find("p", class_="location")
    print(title_element)
    print(company_element)
    print(location_element)
    print()


# In[14]:


for job_element in job_elements:
    title_element = job_element.find("h2", class_="title")
    company_element = job_element.find("h3", class_="company")
    location_element = job_element.find("p", class_="location")
    print(title_element.text)
    print(company_element.text)
    print(location_element.text)
    print()


# In[15]:


python_jobs = results.find_all(
    "h2", string=lambda text: "python" in text.lower()
)


# In[17]:


python_jobs = results.find_all(
    "h2", string=lambda text: "python" in text.lower()
)

python_job_elements = [
    h2_element.parent.parent.parent for h2_element in python_jobs
]


# In[18]:


python_job_elements[0]


# In[19]:


for job_element in python_job_elements:
    # -- snip --
    links = job_element.find_all("a")
    for link in links:
        link_url = link["href"]
        print(f"Apply here: {link_url}\n")


# In[22]:


links[0]['href']


# In[24]:


type(links)

