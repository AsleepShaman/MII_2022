using System;
using System.Collections.Generic;
using System.Data;
using System.Data.Entity;
using System.Linq;
using System.Net;
using System.Web;
using System.Web.Mvc;
using KourseWork.Models;

namespace KoursWork.Controllers
{
    public class Перечень_услугController : Controller
    {
        private AudioStudioDBEntities2 db = new AudioStudioDBEntities2();

        // GET: Перечень_услуг
        public ActionResult Index()
        {
            var перечень_услуг = db.Перечень_услуг.Include(п => п.Аранжировка).Include(п => п.Запись).Include(п => п.Сотрудники).Include(п => п.Сведение);
            return View(перечень_услуг.ToList());
        }
        public ActionResult Error()
        {
            return View();
        }

        // GET: Перечень_услуг/Details/5
        public ActionResult Details(string id)
        {
            if (id == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            Перечень_услуг перечень_услуг = db.Перечень_услуг.Find(id);
            if (перечень_услуг == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            return View(перечень_услуг);
        }

        // GET: Перечень_услуг/Create
        public ActionResult Create()
        {
            ViewBag.Код_услуги = new SelectList(db.Аранжировка, "Код_услуги", "Референс");
            ViewBag.Код_услуги = new SelectList(db.Запись, "Код_услуги", "Что_записывается");
            ViewBag.Паспорт_сотрудника = new SelectList(db.Сотрудники, "Паспорт_сотрудника", "ФИО");
            ViewBag.Код_услуги = new SelectList(db.Сведение, "Код_услуги", "Плагин");
            return View();
        }

        // POST: Перечень_услуг/Create
        // Чтобы защититься от атак чрезмерной передачи данных, включите определенные свойства, для которых следует установить привязку. Дополнительные 
        // сведения см. в разделе https://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult Create([Bind(Include = "Код_услуги,Оборудование,Название,Паспорт_сотрудника")] Перечень_услуг перечень_услуг)
        {
            if (ModelState.IsValid)
            {
                try
                {
                    db.Перечень_услуг.Add(перечень_услуг);
                    db.SaveChanges();
                    return RedirectToAction("Index");
                }
                catch
                {
                    ViewBag.Data = "Error!";
                    return RedirectToAction("Error");
                }
            }

            ViewBag.Код_услуги = new SelectList(db.Аранжировка, "Код_услуги", "Референс", перечень_услуг.Код_услуги);
            ViewBag.Код_услуги = new SelectList(db.Запись, "Код_услуги", "Что_записывается", перечень_услуг.Код_услуги);
            ViewBag.Паспорт_сотрудника = new SelectList(db.Сотрудники, "Паспорт_сотрудника", "ФИО", перечень_услуг.Паспорт_сотрудника);
            ViewBag.Код_услуги = new SelectList(db.Сведение, "Код_услуги", "Плагин", перечень_услуг.Код_услуги);
            return View(перечень_услуг);
        }

        // GET: Перечень_услуг/Edit/5
        public ActionResult Edit(string id)
        {
            if (id == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            Перечень_услуг перечень_услуг = db.Перечень_услуг.Find(id);
            if (перечень_услуг == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            ViewBag.Код_услуги = new SelectList(db.Аранжировка, "Код_услуги", "Референс", перечень_услуг.Код_услуги);
            ViewBag.Код_услуги = new SelectList(db.Запись, "Код_услуги", "Что_записывается", перечень_услуг.Код_услуги);
            ViewBag.Паспорт_сотрудника = new SelectList(db.Сотрудники, "Паспорт_сотрудника", "ФИО", перечень_услуг.Паспорт_сотрудника);
            ViewBag.Код_услуги = new SelectList(db.Сведение, "Код_услуги", "Плагин", перечень_услуг.Код_услуги);
            return View(перечень_услуг);
        }

        // POST: Перечень_услуг/Edit/5
        // Чтобы защититься от атак чрезмерной передачи данных, включите определенные свойства, для которых следует установить привязку. Дополнительные 
        // сведения см. в разделе https://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult Edit([Bind(Include = "Код_услуги,Оборудование,Название,Паспорт_сотрудника")] Перечень_услуг перечень_услуг)
        {
            if (ModelState.IsValid)
            {
                try{db.Entry(перечень_услуг).State = EntityState.Modified;
                db.SaveChanges();
                return RedirectToAction("Index");
                }
                catch
                {
                    ViewBag.Data = "Ошибка редактирования!";
                    return RedirectToAction("Error");
                }
            }
            ViewBag.Код_услуги = new SelectList(db.Аранжировка, "Код_услуги", "Референс", перечень_услуг.Код_услуги);
            ViewBag.Код_услуги = new SelectList(db.Запись, "Код_услуги", "Что_записывается", перечень_услуг.Код_услуги);
            ViewBag.Паспорт_сотрудника = new SelectList(db.Сотрудники, "Паспорт_сотрудника", "ФИО", перечень_услуг.Паспорт_сотрудника);
            ViewBag.Код_услуги = new SelectList(db.Сведение, "Код_услуги", "Плагин", перечень_услуг.Код_услуги);
            return View(перечень_услуг);
        }

        // GET: Перечень_услуг/Delete/5
        public ActionResult Delete(string id)
        {
            if (id == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            Перечень_услуг перечень_услуг = db.Перечень_услуг.Find(id);
            if (перечень_услуг == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            return View(перечень_услуг);
        }

        // POST: Перечень_услуг/Delete/5
        [HttpPost, ActionName("Delete")]
        [ValidateAntiForgeryToken]
        public ActionResult DeleteConfirmed(string id)
        {
            try{Перечень_услуг перечень_услуг = db.Перечень_услуг.Find(id);
            db.Перечень_услуг.Remove(перечень_услуг);
            db.SaveChanges();
            return RedirectToAction("Index");
            }
            catch
            {
                ViewBag.Data = "Ошибка удаления! Возможно, эти данные где-то есть...";
                return RedirectToAction("Error");
            }
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                db.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}
